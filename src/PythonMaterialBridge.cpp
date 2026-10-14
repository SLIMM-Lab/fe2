#include "PythonMaterialBridge.h"
#include "MaterialFactory.h"
#include "declare.h"
#include <jem/base/Exception.h>
#include <jem/base/Thread.h>
#include <jem/io/StringReader.h>
#include <jem/util/Tokenizer.h>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>

namespace
{
std::string text ( const String& value )
{
  return std::string(value.addr(),value.size());
}

class Executor : public jem::Thread
{
 public:
  void run () override
  {
    for (;;)
    {
      std::function<void()> work;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock,[this]{return !queue_.empty();});
        work=std::move(queue_.front());
        queue_.pop();
      }
      work();
    }
  }

  template<class F> auto call ( F work ) -> decltype(work())
  {
    if (jem::Thread::getCurrent()==this) return work();
    using Result=decltype(work());
    auto task=std::make_shared<std::packaged_task<Result()>>([work=std::move(work)] {
      try { return work(); }
      catch (const jem::Exception& error) { throw std::runtime_error(text(error.where())+": "+text(error.what())); }
    });
    auto result=task->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push([task]{(*task)();});
    }
    ready_.notify_one();
    return result.get();
  }
 private:
  std::mutex mutex_;
  std::condition_variable ready_;
  std::queue<std::function<void()>> queue_;
};

Executor& executor ()
{
  // Shared for process lifetime. No Python objects live on this worker, so a
  // Python caller may hold the GIL while waiting for the constitutive result.
  static Executor* instance=[] {
    jem::Thread::getCurrent();
    auto worker=jem::newInstance<Executor>();
    worker->start();
    auto retained=new Ref<Executor>(worker);
    return retained->get();
  }();
  return *instance;
}

void require ( bool condition, const char* message )
{
  if (!condition) throw std::invalid_argument(message);
}

void registerTypes ()
{
  declareModules();
  declareModels();
  declareMaterials();
}
}

struct PythonMaterialFactory::Specification
{
  Properties props, globdat;
  Ref<Material> instantiate ( idx_t count ) const
  {
    Properties conf;
    String type;
    props.getProps("material").get(type,"type");
    require(type!="PRNN","A PRNN cannot recursively be its own physical material layer");
    auto material=newMaterial("material",conf,props,globdat);
    material->configure(props,globdat);
    if (count>0) material->allocPoints(count);
    return material;
  }
};

struct PythonMaterialBatch::State
{
  std::function<Ref<Material>()> instantiate;
  Ref<Material> material;
  idx_t count, components;
  bool pending=false;

  MaterialResponse evaluate ( const Ref<Material>& model, const std::vector<double>& strain ) const
  {
    require(strain.size()==size_t(count*components),"Wrong material strain dimensions");
    require(std::all_of(strain.begin(),strain.end(),[](double x){return std::isfinite(x);}),"Strain must be finite");
    MaterialResponse result;
    result.stress.resize(count*components);
    result.tangent.resize(count*components*components);
    Vector eps(components),sig(components);
    Matrix tangent(components,components);
    for (idx_t p=0;p<count;++p)
    {
      for (idx_t i=0;i<components;++i) eps[i]=strain[p*components+i];
      model->update(sig,tangent,eps,p);
      for (idx_t i=0;i<components;++i)
      {
        result.stress[p*components+i]=sig[i];
        for (idx_t j=0;j<components;++j)
          result.tangent[(p*components+i)*components+j]=tangent(i,j);
      }
    }
    require(std::all_of(result.stress.begin(),result.stress.end(),[](double x){return std::isfinite(x);}),"Nonfinite material stress");
    require(std::all_of(result.tangent.begin(),result.tangent.end(),[](double x){return std::isfinite(x);}),"Nonfinite material tangent");
    return result;
  }

  std::vector<double> replay ( const std::vector<double>& history, idx_t steps,
                               const std::vector<double>& strain ) const
  {
    auto model=instantiate();
    const idx_t width=count*components;
    for (idx_t t=0;t<steps;++t)
    {
      std::vector<double> eps(history.begin()+t*width,history.begin()+(t+1)*width);
      evaluate(model,eps);
      model->commit();
    }
    return evaluate(model,strain).stress;
  }
};

PythonMaterialFactory::PythonMaterialFactory ( const std::string& properties )
{
  auto raw=executor().call([&] {
    registerTypes();
    auto spec=std::make_unique<Specification>();
    auto reader=jem::newInstance<jem::io::StringReader>(String(properties.c_str()));
    auto tokenizer=jem::newInstance<jem::util::Tokenizer>(reader);
    spec->props.parseFrom(*tokenizer,"Python material configuration");
    spec->instantiate(0); // Validate settings without creating online history.
    return spec.release();
  });
  spec_=std::shared_ptr<Specification>(raw,[](Specification* p){executor().call([p]{delete p;});});
}

PythonMaterialFactory::PythonMaterialFactory ( const Properties& materialProps, const Properties& globdat )
{
  auto raw=executor().call([&] {
    registerTypes();
    auto spec=std::make_unique<Specification>();
    spec->props.set("material",materialProps);
    spec->globdat=globdat;
    spec->instantiate(0);
    return spec.release();
  });
  spec_=std::shared_ptr<Specification>(raw,[](Specification* p){executor().call([p]{delete p;});});
}

PythonMaterialFactory::~PythonMaterialFactory () = default;

std::shared_ptr<PythonMaterialBatch> PythonMaterialFactory::create ( idx_t count ) const
{
  require(count>0,"Material point count must be positive");
  auto state=executor().call([&] {
    auto value=std::make_unique<PythonMaterialBatch::State>();
    value->instantiate=[spec=spec_,count]{return spec->instantiate(count);};
    value->material=value->instantiate();
    value->count=count;
    auto rank=value->material->rank();
    require(rank>=1 && rank<=3,"Unsupported material rank");
    value->components=FEMUtils::STRAIN_COUNTS[rank];
    return value.release();
  });
  return std::shared_ptr<PythonMaterialBatch>(new PythonMaterialBatch(state,count,state->components));
}

std::vector<std::string> PythonMaterialFactory::types ()
{
  return executor().call([] {
    registerTypes();
    auto names=MaterialFactory::listKnownTypes();
    std::vector<std::string> result;
    for (idx_t i=0;i<names.size();++i) result.push_back(text(names[i]));
    return result;
  });
}

PythonMaterialBatch::PythonMaterialBatch ( State* state, idx_t count, idx_t components )
  : state_(state),count_(count),components_(components) {}
PythonMaterialBatch::~PythonMaterialBatch () { executor().call([this]{delete state_;}); }
idx_t PythonMaterialBatch::count () const noexcept {return count_;}
idx_t PythonMaterialBatch::components () const noexcept {return components_;}

MaterialResponse PythonMaterialBatch::update ( const std::vector<double>& strain )
{
  return executor().call([&] {
    try
    {
      auto result=state_->evaluate(state_->material,strain);
      state_->pending=true;
      return result;
    }
    catch (...) {state_->material->cancel();state_->pending=false;throw;}
  });
}

void PythonMaterialBatch::commit ()
{
  executor().call([this] {
    if (state_->pending) state_->material->commit();
    state_->pending=false;
  });
}

void PythonMaterialBatch::cancel ()
{
  executor().call([this] {state_->material->cancel();state_->pending=false;});
}

std::vector<double> PythonMaterialBatch::historyVjp ( const std::vector<double>& history,
    idx_t steps, const std::vector<double>& strain, const std::vector<double>& adjoint, double step )
{
  require(steps>=0 && history.size()==size_t(steps*count_*components_),"Wrong material history dimensions");
  require(adjoint.size()==size_t(count_*components_),"Wrong stress adjoint dimensions");
  require(step>0 && std::isfinite(step),"Finite-difference step must be positive and finite");
  return executor().call([&] {
    // Isolated replicas: differentiation never advances the live material.
    std::vector<double> result(history.size()),perturbed=history;
    for (size_t i=0;i<history.size();++i)
    {
      double h=step*std::max(1.0,std::abs(history[i]));
      perturbed[i]=history[i]+h;
      auto plus=state_->replay(perturbed,steps,strain);
      perturbed[i]=history[i]-h;
      auto minus=state_->replay(perturbed,steps,strain);
      perturbed[i]=history[i];
      for (size_t j=0;j<adjoint.size();++j) result[i]+=(plus[j]-minus[j])*adjoint[j]/(2*h);
    }
    return result;
  });
}
