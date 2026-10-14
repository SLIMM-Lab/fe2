#include "PRNNMaterial.h"
#include "PythonMaterialBridge.h"
#include "MaterialFactory.h"
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalOperationException.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <mutex>
#include <vector>

namespace py=pybind11;
JEM_DEFINE_CLASS ( PRNNMaterial );
const char* PRNNMaterial::TYPE_NAME="PRNN";

namespace
{
std::string text ( const String& value ) {return std::string(value.addr(),value.size());}

void initializePython ()
{
  static std::once_flag once;
  std::call_once(once,[] {
    if (!Py_IsInitialized())
    {
      PyConfig config;
      PyConfig_InitPythonConfig(&config);
      PyStatus status=PyConfig_SetBytesString(&config,&config.program_name,FE2_PYTHON_EXECUTABLE);
      if (PyStatus_Exception(status)) throw std::runtime_error("Cannot configure embedded Python executable");
      py::initialize_interpreter(&config);
      PyEval_SaveThread();
    }
  });
  // Interpreter lives for the host process. Never finalize PyTorch underneath
  // another material or a running Jem worker.
}

py::array_t<double> input ( const Vector& strain )
{
  py::array_t<double> result(static_cast<py::ssize_t>(strain.size()));
  for (idx_t i=0;i<strain.size();++i) result.mutable_data()[i]=strain[i];
  return result;
}
}

class __attribute__((visibility("hidden"))) PRNNMaterial::Impl
{
 public:
  idx_t rank=0,components=0;
  ProblemType state=ProblemType::PlaneStrain;
  String stateName,weights,module="pyprnn.checkpoint",loader="load_model",pythonPath="";
  std::shared_ptr<PythonMaterialFactory> factory;
  py::object loadModel,runtime;
  std::vector<py::object> models;
  Properties materialProps;
  bool configured=false;
};

PRNNMaterial::PRNNMaterial ( const String& name,const Properties& conf,
    const Properties& props,const Properties& ) : Super(name),impl_(new Impl)
{
  props.getProps(name).get(impl_->rank,RANK_PROP,idx_t(2),idx_t(3));
  conf.makeProps(name).set(RANK_PROP,impl_->rank);
  impl_->components=FEMUtils::STRAIN_COUNTS[impl_->rank];
}

PRNNMaterial::~PRNNMaterial ()
{
  if (Py_IsInitialized()) {py::gil_scoped_acquire gil;impl_.reset();}
}

void PRNNMaterial::configure ( const Properties& props,const Properties& globdat )
{
  if (!impl_->models.empty())
    throw jem::IllegalOperationException(getContext(),"Cannot reconfigure allocated Python PRNN history");
  auto p=props.getProps(myName_);
  p.get(impl_->weights,"weights");
  p.find(impl_->module,"module");
  p.find(impl_->loader,"loader");
  p.find(impl_->pythonPath,"pythonPath");
  impl_->stateName="Solid";
  if (impl_->rank==2) p.get(impl_->stateName,"state");
  if (impl_->rank==2 && impl_->stateName=="PlaneStress") impl_->state=ProblemType::PlaneStress;
  else if (impl_->rank==2 && impl_->stateName=="PlaneStrain") impl_->state=ProblemType::PlaneStrain;
  else if (impl_->rank==3 && impl_->stateName=="Solid") impl_->state=ProblemType::Solid;
  else throw jem::IllegalInputException(getContext(),"Invalid macro rank/state");
  impl_->materialProps=p.getProps("material");
  impl_->factory=std::make_shared<PythonMaterialFactory>(impl_->materialProps,globdat);
  initializePython();
  py::gil_scoped_acquire gil;
  try
  {
    auto path=py::module_::import("sys").attr("path");
    path.attr("insert")(0,FE2_PYTHON_SOURCE);
    if (impl_->pythonPath.size()) path.attr("insert")(0,text(impl_->pythonPath));
    py::module_::import("_fe2_material");
    impl_->runtime=py::module_::import("pyprnn.runtime");
    impl_->loadModel=py::module_::import(text(impl_->module).c_str()).attr(text(impl_->loader).c_str());
    impl_->configured=true;
  }
  catch (const py::error_already_set& error) {throw jem::IllegalInputException(getContext(),String(error.what()));}
}

void PRNNMaterial::getConfig ( const Properties& conf,const Properties& ) const
{
  auto p=conf.makeProps(myName_);
  p.set(RANK_PROP,impl_->rank);
  p.set("state",impl_->stateName);
  p.set("weights",impl_->weights);
  p.set("module",impl_->module);
  p.set("loader",impl_->loader);
  p.set("pythonPath",impl_->pythonPath);
  p.makeProps("material").mergeWith(impl_->materialProps);
}

void PRNNMaterial::allocPoints ( idx_t count )
{
  JEM_PRECHECK(impl_->configured && count>=0);
  py::gil_scoped_acquire gil;
  try
  {
    for (idx_t i=0;i<count;++i)
      impl_->models.push_back(impl_->loadModel(text(impl_->weights),py::cast(impl_->factory)));
  }
  catch (const py::error_already_set& error) {throw jem::IllegalInputException(getContext(),String(error.what()));}
}

void PRNNMaterial::commit ()
{
  py::gil_scoped_acquire gil;
  try {for (auto& model:impl_->models) model.attr("commit")();}
  catch (const py::error_already_set& error) {throw jem::IllegalOperationException(getContext(),String(error.what()));}
}

void PRNNMaterial::cancel ()
{
  py::gil_scoped_acquire gil;
  try {for (auto& model:impl_->models) model.attr("cancel")();}
  catch (const py::error_already_set& error) {throw jem::IllegalOperationException(getContext(),String(error.what()));}
}

idx_t PRNNMaterial::rank () const noexcept {return impl_->rank;}
ProblemType PRNNMaterial::getProblemType () const noexcept {return impl_->state;}

void PRNNMaterial::update ( const Vector& stress,const Matrix& stiff,const Vector& strain,idx_t ip ) const
{
  JEM_PRECHECK(ip>=0 && ip<idx_t(impl_->models.size()) && strain.size()==impl_->components &&
               stress.size()==impl_->components && stiff.size(0)==impl_->components && stiff.size(1)==impl_->components);
  py::gil_scoped_acquire gil;
  try
  {
    auto result=impl_->runtime.attr("evaluate_with_tangent")(impl_->models[ip],input(strain)).cast<py::tuple>();
    auto sig=result[0].cast<py::array_t<double>>();
    auto c=result[1].cast<py::array_t<double>>();
    if (sig.ndim()!=1 || sig.shape(0)!=impl_->components || c.ndim()!=2 ||
        c.shape(0)!=impl_->components || c.shape(1)!=impl_->components)
      throw jem::IllegalInputException(getContext(),"Invalid Python response dimensions");
    auto s=sig.unchecked<1>();auto k=c.unchecked<2>();
    for (idx_t i=0;i<impl_->components;++i)
    {
      stress[i]=s(i);
      for (idx_t j=0;j<impl_->components;++j) stiff(i,j)=k(i,j);
    }
  }
  catch (const py::error_already_set& error)
  {
    impl_->models[ip].attr("cancel")();
    throw jem::IllegalOperationException(getContext(),String(error.what()));
  }
}

void PRNNMaterial::update ( const Vector& stress,const Vector& strain,idx_t ip ) const
{
  JEM_PRECHECK(ip>=0 && ip<idx_t(impl_->models.size()) && strain.size()==impl_->components && stress.size()==impl_->components);
  py::gil_scoped_acquire gil;
  try
  {
    auto result=impl_->runtime.attr("observe")(impl_->models[ip],input(strain)).cast<py::array_t<double>>();
    if (result.ndim()!=1 || result.shape(0)!=impl_->components)
      throw jem::IllegalInputException(getContext(),"Invalid Python stress dimensions");
    auto s=result.unchecked<1>();
    for (idx_t i=0;i<impl_->components;++i) stress[i]=s(i);
  }
  catch (const py::error_already_set& error) {throw jem::IllegalOperationException(getContext(),String(error.what()));}
}

void PRNNMaterial::elasticUpdate ( const Vector&,const Matrix&,const Vector& ) const
{throw jem::IllegalOperationException(getContext(),"PRNN has no separate elastic constitutive law");}
Vector PRNNMaterial::elasticUpdate ( const Vector& ) const
{throw jem::IllegalOperationException(getContext(),"PRNN has no separate elastic constitutive law");}
Matrix PRNNMaterial::getStiffMat () const
{throw jem::IllegalOperationException(getContext(),"Use the tangent returned by PRNN update");}
Tuple<double,6> PRNNMaterial::fill3DStrain ( const Vector& v ) const
{
  if (impl_->rank==3) return FEMUtils::fillFrom3D(v);
  if (impl_->state==ProblemType::PlaneStrain) return FEMUtils::fillFrom2D(v,0.0);
  throw jem::IllegalOperationException(getContext(),"Out-of-plane strain is not a network output");
}
Tuple<double,6> PRNNMaterial::fill3DStress ( const Vector& v ) const
{
  if (impl_->rank==3) return FEMUtils::fillFrom3D(v);
  if (impl_->state==ProblemType::PlaneStress) return FEMUtils::fillFrom2D(v,0.0);
  throw jem::IllegalOperationException(getContext(),"Out-of-plane stress is not a network output");
}
Ref<Material> PRNNMaterial::makeNew ( const String& name,const Properties& conf,
    const Properties& props,const Properties& globdat )
{return jem::newInstance<Self>(name,conf,props,globdat);}
void PRNNMaterial::declare () {MaterialFactory::declare(TYPE_NAME,&makeNew);}
