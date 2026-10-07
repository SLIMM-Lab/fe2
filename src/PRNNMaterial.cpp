#include "PRNNMaterial.h"
#include "J2Material.h"
#include "MaterialFactory.h"
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalOperationException.h>
#include <torch/script.h>
#include <mutex>
#include <vector>

JEM_DEFINE_CLASS ( PRNNMaterial );
const char* PRNNMaterial::TYPE_NAME = "PRNN";

class PRNNMaterial::Impl
{
 public:
  struct Point
  {
    Ref<J2Material> material;
    Vector trialStrain;
    bool dirty = false;
    std::mutex mutex;
  };

  idx_t rank = 0, components = 0, latentPoints = 0;
  ProblemType state = ProblemType::PlaneStrain;
  String stateName, weights;
  Properties localProps, globdat;
  Ref<J2Material> prototype;
  at::Tensor encoder, decoder, scale, offset;
  Matrix elastic;
  std::vector<std::unique_ptr<Point>> points;
  bool configured = false;

  Ref<J2Material> makeMaterial () const
  {
    Properties conf;
    auto material = jem::newInstance<J2Material>
      ( "local", conf, localProps, globdat );
    material->configure ( localProps, globdat );
    material->allocPoints ( latentPoints );
    return material;
  }

  // No configure/commit here. Every trial is evaluated from J2's preHist_.
  void evaluate ( const Ref<J2Material>& material, const Vector& stress,
                  const Matrix& stiff, const Vector& strain ) const
  {
    at::NoGradGuard noGrad;
    auto input = at::empty ( {components}, encoder.options() );
    for ( idx_t i = 0; i < components; ++i )
      input.data_ptr<double>()[i] = strain[i];
    auto local = at::mv ( encoder, input * scale + offset ).contiguous();
    auto localStress = at::empty_like ( local );
    auto blocks = at::zeros ( {components * latentPoints, components}, encoder.options() );
    Vector eps ( components ), sig ( components );
    Matrix tangent ( components, components );
    for ( idx_t p = 0; p < latentPoints; ++p )
    {
      for ( idx_t i = 0; i < components; ++i )
        eps[i] = local.data_ptr<double>()[p * components + i];
      material->update ( sig, tangent, eps, p );
      for ( idx_t i = 0; i < components; ++i )
      {
        localStress.data_ptr<double>()[p * components + i] = sig[i];
        for ( idx_t j = 0; j < components; ++j )
          for ( idx_t k = 0; k < components; ++k )
            blocks.data_ptr<double>()[(p * components + i) * components + j] +=
              tangent(i,k) * encoder.data_ptr<double>()[(p * components + k) * components + j]
              * scale.data_ptr<double>()[j];
      }
    }
    auto result = at::mv ( decoder, localStress ).contiguous();
    auto jacobian = at::mm ( decoder, blocks ).contiguous();
    TORCH_CHECK ( at::isfinite(result).all().item<bool>() &&
                  at::isfinite(jacobian).all().item<bool>(), "Nonfinite PRNN response" );
    for ( idx_t i = 0; i < components; ++i )
    {
      stress[i] = result.data_ptr<double>()[i];
      for ( idx_t j = 0; j < components; ++j )
        stiff(i,j) = jacobian.data_ptr<double>()[i * components + j];
    }
  }
};

PRNNMaterial::PRNNMaterial ( const String& name, const Properties& conf,
                            const Properties& props, const Properties& )
  : Super ( name ), impl_ ( new Impl )
{
  props.getProps(name).get ( impl_->rank, RANK_PROP, idx_t(2), idx_t(3) );
  conf.makeProps(name).set ( RANK_PROP, impl_->rank );
  impl_->components = impl_->rank == 2 ? 3 : 6;
}

PRNNMaterial::~PRNNMaterial () = default;

void PRNNMaterial::configure ( const Properties& props, const Properties& globdat )
{
  at::NoGradGuard noGrad;
  if ( !impl_->points.empty() )
    throw jem::IllegalOperationException ( getContext(), "Cannot reconfigure allocated PRNN history" );
  auto myProps = props.getProps ( myName_ );
  myProps.get ( impl_->weights, "weights" );
  impl_->stateName = "Solid";
  if ( impl_->rank == 2 ) myProps.get ( impl_->stateName, "state" );
  if ( impl_->stateName == "PlaneStrain" && impl_->rank == 2 )
    impl_->state = ProblemType::PlaneStrain;
  else if ( impl_->stateName == "PlaneStress" && impl_->rank == 2 )
    impl_->state = ProblemType::PlaneStress;
  else if ( impl_->stateName == "Solid" && impl_->rank == 3 )
    impl_->state = ProblemType::Solid;
  else
    throw jem::IllegalInputException ( getContext(), "Invalid PRNN rank/state combination" );

  try
  {
    auto archive = torch::jit::load ( std::string(impl_->weights.addr(), impl_->weights.size()), at::kCPU );
    TORCH_CHECK ( archive.attr("format_version").toTensor().item<int64_t>() == 1,
                  "Unsupported PRNN weight archive version" );
    auto options = at::TensorOptions().device(at::kCPU).dtype(at::kDouble);
    impl_->encoder = archive.attr("encoder_weight").toTensor().detach().to(options).contiguous();
    auto rawDecoder = archive.attr("decoder_weight").toTensor().detach().to(options);
    impl_->decoder = at::softplus ( rawDecoder ).contiguous();
    impl_->scale = archive.attr("strain_scale").toTensor().detach().to(options).contiguous();
    impl_->offset = archive.attr("strain_offset").toTensor().detach().to(options).contiguous();
    auto& e = impl_->encoder;
    auto& d = impl_->decoder;
    idx_t n = impl_->components;
    TORCH_CHECK ( e.dim() == 2 && d.dim() == 2 && e.size(1) == n &&
                  e.size(0) > 0 && e.size(0) % n == 0 &&
                  d.size(0) == n && d.size(1) == e.size(0), "Invalid encoder/decoder dimensions" );
    TORCH_CHECK ( impl_->scale.sizes() == at::IntArrayRef({n}) &&
                  impl_->offset.sizes() == at::IntArrayRef({n}), "Invalid strain normalization dimensions" );
    TORCH_CHECK ( at::isfinite(e).all().item<bool>() && at::isfinite(rawDecoder).all().item<bool>() &&
                  at::isfinite(impl_->scale).all().item<bool>() &&
                  (impl_->scale > 0).all().item<bool>() && at::isfinite(impl_->offset).all().item<bool>(),
                  "Invalid/nonfinite PRNN weights or strain normalization" );
    impl_->latentPoints = e.size(0) / n;
  }
  catch ( const c10::Error& error )
  {
    throw jem::IllegalInputException ( getContext(), String(error.what()) );
  }

  // Reuse the existing J2 implementation and its supported property schema.
  impl_->localProps = Properties();
  auto local = impl_->localProps.makeProps ( "local" );
  auto materialProps = myProps.getProps ( "material" );
  // Read through the original Properties so Jive's configuration snooper sees
  // which J2 options were consumed, while synthesizing rank/state separately.
  for ( const char* key : {"young", "poisson", "yield", "rmTolerance", "rmMaxIter", "args", "rho"} )
  {
    Ref<jem::Object> value;
    if ( materialProps.find(value,key) ) local.set(key,value);
  }
  local.set ( RANK_PROP, impl_->rank );
  local.set ( "state", impl_->stateName );
  impl_->globdat = globdat;
  impl_->prototype = impl_->makeMaterial();
  auto c = impl_->prototype->getStiffMat();
  auto blocks = at::zeros ( {impl_->latentPoints * impl_->components, impl_->components}, impl_->encoder.options() );
  for ( idx_t p = 0; p < impl_->latentPoints; ++p )
    for ( idx_t i = 0; i < impl_->components; ++i )
      for ( idx_t j = 0; j < impl_->components; ++j )
        for ( idx_t k = 0; k < impl_->components; ++k )
          blocks.data_ptr<double>()[(p * impl_->components + i) * impl_->components + j] +=
            c(i,k) * impl_->encoder.data_ptr<double>()[(p * impl_->components + k) * impl_->components + j]
            * impl_->scale.data_ptr<double>()[j];
  auto elastic = at::mm ( impl_->decoder, blocks ).contiguous();
  impl_->elastic.resize ( impl_->components, impl_->components );
  for ( idx_t i = 0; i < impl_->components; ++i )
    for ( idx_t j = 0; j < impl_->components; ++j )
      impl_->elastic(i,j) = elastic.data_ptr<double>()[i * impl_->components + j];
  impl_->configured = true;
}

void PRNNMaterial::getConfig ( const Properties& conf, const Properties& globdat ) const
{
  auto myConf = conf.makeProps ( myName_ );
  myConf.set ( RANK_PROP, impl_->rank );
  myConf.set ( "state", impl_->stateName );
  myConf.set ( "weights", impl_->weights );
  if ( impl_->configured )
  {
    Properties localConf;
    impl_->prototype->getConfig ( localConf, globdat );
    myConf.makeProps("material").mergeWith ( localConf.getProps("local") );
  }
}

void PRNNMaterial::allocPoints ( idx_t count )
{
  JEM_PRECHECK ( impl_->configured && count >= 0 );
  for ( idx_t i = 0; i < count; ++i )
  {
    auto point = std::make_unique<Impl::Point>();
    point->material = impl_->makeMaterial();
    point->trialStrain.resize ( impl_->components );
    impl_->points.push_back ( std::move(point) );
  }
}

void PRNNMaterial::commit ()
{
  for ( auto& point : impl_->points )
  {
    std::lock_guard<std::mutex> lock ( point->mutex );
    if ( point->dirty ) point->material->commit();
    point->material->cancel();
    point->dirty = false;
  }
}

void PRNNMaterial::cancel ()
{
  for ( auto& point : impl_->points )
  {
    std::lock_guard<std::mutex> lock ( point->mutex );
    point->material->cancel();
    point->dirty = false;
  }
}

idx_t PRNNMaterial::rank () const noexcept { return impl_->rank; }
ProblemType PRNNMaterial::getProblemType () const noexcept { return impl_->state; }

void PRNNMaterial::update ( const Vector& stress, const Matrix& stiff,
                            const Vector& strain, idx_t ip ) const
{
  JEM_PRECHECK ( ip >= 0 && ip < idx_t(impl_->points.size()) &&
                 strain.size() == impl_->components && stress.size() == impl_->components &&
                 stiff.size(0) == impl_->components && stiff.size(1) == impl_->components );
  auto& point = *impl_->points[ip];
  std::lock_guard<std::mutex> lock ( point.mutex );
  try
  {
    impl_->evaluate ( point.material, stress, stiff, strain );
    point.trialStrain = strain;
    point.dirty = true;
  }
  catch (...)
  {
    point.material->cancel();
    point.dirty = false;
    throw;
  }
}

// GET_TABLE is observational: it must not replace an outstanding Newton trial.
void PRNNMaterial::update ( const Vector& stress, const Vector& strain, idx_t ip ) const
{
  JEM_PRECHECK ( ip >= 0 && ip < idx_t(impl_->points.size()) &&
                 strain.size() == impl_->components && stress.size() == impl_->components );
  auto& point = *impl_->points[ip];
  std::lock_guard<std::mutex> lock ( point.mutex );
  Matrix tangent ( impl_->components, impl_->components );
  Vector ignored ( impl_->components );
  try
  {
    impl_->evaluate ( point.material, stress, tangent, strain );
    if ( point.dirty )
      impl_->evaluate ( point.material, ignored, tangent, point.trialStrain );
    else
      point.material->cancel();
  }
  catch (...)
  {
    point.material->cancel();
    point.dirty = false;
    throw;
  }
}

Matrix PRNNMaterial::getStiffMat () const
{
  JEM_PRECHECK ( impl_->configured );
  return impl_->elastic.clone();
}

void PRNNMaterial::elasticUpdate ( const Vector& stress, const Matrix& stiff,
                                   const Vector& strain ) const
{
  JEM_PRECHECK ( impl_->configured );
  // Evaluate virgin elastic response, including an optional normalization offset.
  at::NoGradGuard noGrad;
  auto c = impl_->prototype->getStiffMat();
  auto input = at::empty ( {impl_->components}, impl_->encoder.options() );
  for ( idx_t i = 0; i < impl_->components; ++i ) input.data_ptr<double>()[i] = strain[i];
  auto eps = at::mv ( impl_->encoder, input * impl_->scale + impl_->offset ).contiguous();
  auto sig = at::empty_like ( eps );
  for ( idx_t p = 0; p < impl_->latentPoints; ++p )
    for ( idx_t i = 0; i < impl_->components; ++i )
    {
      double value = 0;
      for ( idx_t j = 0; j < impl_->components; ++j )
        value += c(i,j) * eps.data_ptr<double>()[p * impl_->components + j];
      sig.data_ptr<double>()[p * impl_->components + i] = value;
    }
  auto result = at::mv ( impl_->decoder, sig ).contiguous();
  for ( idx_t i = 0; i < impl_->components; ++i ) stress[i] = result.data_ptr<double>()[i];
  stiff = impl_->elastic;
}

Vector PRNNMaterial::elasticUpdate ( const Vector& strain ) const
{
  Vector stress ( impl_->components );
  Matrix tangent ( impl_->components, impl_->components );
  elasticUpdate ( stress, tangent, strain );
  return stress;
}

Tuple<double,6> PRNNMaterial::fill3DStrain ( const Vector& strain ) const
{
  if ( impl_->rank == 3 ) return FEMUtils::fillFrom3D ( strain );
  if ( impl_->state == ProblemType::PlaneStrain ) return FEMUtils::fillFrom2D ( strain, 0.0 );
  throw jem::IllegalOperationException ( getContext(), "Out-of-plane strain is not a PRNN output" );
}

Tuple<double,6> PRNNMaterial::fill3DStress ( const Vector& stress ) const
{
  if ( impl_->rank == 3 ) return FEMUtils::fillFrom3D ( stress );
  if ( impl_->state == ProblemType::PlaneStress ) return FEMUtils::fillFrom2D ( stress, 0.0 );
  throw jem::IllegalOperationException ( getContext(), "Out-of-plane stress is not a PRNN output" );
}

Ref<Material> PRNNMaterial::makeNew ( const String& name, const Properties& conf,
                                    const Properties& props, const Properties& globdat )
{
  return jem::newInstance<Self> ( name, conf, props, globdat );
}

void PRNNMaterial::declare ()
{
  MaterialFactory::declare ( TYPE_NAME, &makeNew );
}
