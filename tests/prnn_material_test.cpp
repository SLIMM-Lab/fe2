#include "PRNNMaterial.h"
#include "J2Material.h"
#include <jem/base/Exception.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <jem/base/Thread.h>
#include <functional>
#include <vector>

// Jem's allocator requires Jem-managed threads (as used by SolidModel).
class Worker : public jem::Thread
{
 public:
  explicit Worker ( std::function<void()> work ) : work_(std::move(work)) {}
  void run () override { work_(); }
 private:
  std::function<void()> work_;
};

static void require ( bool condition, const char* message )
{
  if ( !condition ) throw std::runtime_error ( message );
}

static double normDiff ( const Vector& a, const Vector& b )
{
  double error = 0;
  for ( idx_t i = 0; i < a.size(); ++i ) error = std::max(error, std::abs(a[i] - b[i]));
  return error;
}

static Properties properties ( const char* weights, idx_t rank, const char* state )
{
  Properties props;
  auto p = props.makeProps("network");
  p.set("rank", rank);
  p.set("state", state);
  p.set("weights", weights);
  auto m = p.makeProps("material");
  m.set("young", 3130.0);
  m.set("poisson", 0.37);
  m.set("yield", "64.80 - 33.60 * exp(x / -0.003407)");
  m.set("rmTolerance", 1.e-12);
  m.set("rmMaxIter", idx_t(100));
  return props;
}

static Ref<PRNNMaterial> network ( const Properties& props, idx_t points = 2 )
{
  Properties conf, globdat;
  auto m = jem::newInstance<PRNNMaterial>("network", conf, props, globdat);
  m->configure(props, globdat);
  m->allocPoints(points);
  return m;
}

static Vector strain ( idx_t n, double value )
{
  Vector e(n);
  e = 0.0;
  e[0] = value;
  e[1] = -0.21 * value;
  e[n == 3 ? 2 : 3] = 0.37 * value;
  return e;
}

static void tangentCheck ( const Ref<PRNNMaterial>& m, const Vector& e )
{
  idx_t n = e.size();
  Vector s(n), plus(n), minus(n);
  Matrix c(n,n), scratch(n,n);
  m->update(s,c,e,0);
  double worst = 0;
  for ( idx_t j = 0; j < n; ++j )
  {
    Vector ep = e.clone(), em = e.clone();
    const double h = 1.e-7;
    ep[j] += h;
    em[j] -= h;
    m->update(plus,scratch,ep,0);
    m->update(minus,scratch,em,0);
    for ( idx_t i = 0; i < n; ++i )
    {
      double fd = (plus[i] - minus[i]) / (2*h);
      double error = std::abs(c(i,j)-fd) / std::max(1.0, std::abs(fd));
      worst = std::max(worst,error);
    }
  }
  std::cout << "tangent relative error: " << worst << '\n';
  require(worst < 2.e-5, "PRNN tangent disagrees with finite differences");
  // Leave the unperturbed trial as the candidate for commit.
  m->update(s,c,e,0);
}

static void run ( const char* weights, idx_t rank, const char* state )
{
  idx_t n = rank == 2 ? 3 : 6;
  auto props = properties(weights,rank,state);
  auto m = network(props), reference = network(props);
  Matrix c(n,n);
  Vector a(n), b(n), virgin(n);
  auto elastic = strain(n,0.0001), plastic = strain(n,0.08), unload = strain(n,0.013);
  tangentCheck(m,elastic);
  tangentCheck(m,plastic);
  m->update(a,c,plastic,0);
  m->update(b,c,plastic,0);
  require(normDiff(a,b) < 1.e-10, "Repeated Newton trial advances history");
  m->update(b,c,unload,0);
  m->update(b,c,plastic,0);
  require(normDiff(a,b) < 1.e-10, "Newton trial depends on previous trial");
  // Stress-only output at a different strain must preserve the pending trial.
  m->update(b,unload,0);
  reference->update(b,c,plastic,0);
  m->commit();
  reference->commit();
  m->commit(); // no dirty point: never swap an old history back in
  m->update(a,c,unload,0);
  reference->update(b,c,unload,0);
  require(normDiff(a,b) < 1.e-9, "Commit/GET_TABLE corrupts pending state");
  m->update(virgin,c,unload,1);
  require(normDiff(a,virgin) > 1.0, "History is reset at forward or shared between macro points");
  tangentCheck(m,unload);
  // Cancel and even a commit after cancel must retain only accepted history.
  m->update(a,c,strain(n,0.2),0);
  m->cancel();
  m->commit();
  m->update(a,c,unload,0);
  require(normDiff(a,b) < 1.e-9, "Cancel accepts rejected plastic history");
  // Output evaluation without a Newton trial must not create a committable state.
  m->cancel();
  m->update(virgin,strain(n,0.4),0);
  m->commit();
  m->update(a,c,unload,0);
  require(normDiff(a,b) < 1.e-9, "Read-only output modifies history");
  // Distinct macro points can use independent J2 return-map workspaces in parallel.
  m->cancel();
  std::vector<Vector> outputs;
  for (int i=0;i<8;++i) outputs.push_back(Vector(n));
  m->allocPoints(6);
  std::vector<Ref<Worker>> threads;
  for (int i=1;i<8;++i)
  {
    auto thread = jem::newInstance<Worker>([&,i] {
      Matrix k(n,n);
      for (int repeat=0;repeat<10;++repeat) m->update(outputs[i],k,plastic,i);
    });
    thread->start();
    threads.push_back(thread);
  }
  for (auto& thread : threads) thread->join();
  for (int i=2;i<8;++i) require(normDiff(outputs[1],outputs[i]) < 1.e-10, "Parallel history race");
  std::cout << state << ": lifecycle, tangent and parallel checks passed\n";
}

int main ( int argc, char** argv )
{
  try
  {
    if (argc == 3 && std::string(argv[1]) == "--check-checkpoint")
    {
      run(argv[2],2,"PlaneStress");
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--replay")
    {
      auto m = network(properties(argv[2],2,"PlaneStress"),1);
      Vector eps(3), sig(3);
      Matrix c(3,3);
      std::cout << std::setprecision(17);
      while (std::cin >> eps[0] >> eps[1] >> eps[2])
      {
        m->update(sig,c,eps,0);
        m->commit();
        std::cout << sig[0] << ' ' << sig[1] << ' ' << sig[2] << '\n';
      }
      return 0;
    }
    require(argc == 4,"Pass fixture3.pt fixture6.pt normalized.pt");
    run(argv[1],2,"PlaneStress");
    run(argv[1],2,"PlaneStrain");
    run(argv[2],3,"Solid");
    run(argv[3],2,"PlaneStress");
    bool rejectedDimensions = false;
    try { network(properties(argv[1],3,"Solid")); }
    catch ( const jem::Exception& ) { rejectedDimensions = true; }
    require(rejectedDimensions,"Checkpoint dimensions must match material rank");
    // Validate latent ordering and SoftLayer mapping independently of PRNN code.
    auto props = properties(argv[1],2,"PlaneStress");
    auto net = network(props);
    Properties jprops, conf, globdat;
    jprops.makeProps("local").mergeWith(props.getProps("network.material"));
    jprops.makeProps("local").set("rank",idx_t(2));
    jprops.makeProps("local").set("state","PlaneStress");
    auto j2 = jem::newInstance<J2Material>("local",conf,jprops,globdat);
    j2->configure(jprops,globdat);
    j2->allocPoints(2);
    Vector e = strain(3,0.05), s(3), expected(3), local(3), sig(3);
    Matrix c(3,3);
    expected = 0.0;
    for (idx_t p=0;p<2;++p)
    {
      for (idx_t i=0;i<3;++i) local[i] = e[i] * (p==0 ? 1.0 : 0.7);
      if (p==0) local[0] += 0.12 * e[1];
      else local[1] -= 0.09 * e[0];
      j2->update(sig,c,local,p);
      for (idx_t i=0;i<3;++i)
        for (idx_t j=0;j<3;++j)
        {
          double raw = i==j ? (p==0 ? 0.4 : -0.2) : -3.0;
          expected[i] += std::log1p(std::exp(raw)) * sig[j];
        }
    }
    net->update(s,c,e,0);
    require(normDiff(s,expected) < 1.e-9,"Encoder, latent ordering or softplus decoder mismatch");
    std::cout << "All PRNN material checks passed\n";
    return 0;
  }
  catch ( const jem::Exception& error )
  {
    auto message = error.what();
    std::cerr << std::string(message.addr(),message.size()) << '\n';
  }
  catch ( const std::exception& error )
  {
    std::cerr << error.what() << '\n';
  }
  return 1;
}
