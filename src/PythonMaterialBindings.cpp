#include "PythonMaterialBridge.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
using Input=py::array_t<double,py::array::c_style|py::array::forcecast>;

static std::vector<double> values ( const Input& array )
{
  return std::vector<double>(array.data(),array.data()+array.size());
}

static py::array_t<double> array ( const std::vector<double>& data,
                                  std::initializer_list<py::ssize_t> shape )
{
  py::array_t<double> result(shape);
  if (!data.empty()) std::memcpy(result.mutable_data(),data.data(),data.size()*sizeof(double));
  return result;
}

static void bindMaterials ( py::module_& module )
{
  module.doc()="Generic Jem/Jive material factory, trial updates and history derivatives";
  py::class_<PythonMaterialBatch,std::shared_ptr<PythonMaterialBatch>>(module,"MaterialBatch")
    .def_property_readonly("count",[](const PythonMaterialBatch& m){return int(m.count());})
    .def_property_readonly("components",[](const PythonMaterialBatch& m){return int(m.components());})
    .def("update",[](PythonMaterialBatch& material,const Input& strain) {
      if (strain.ndim()!=2 || strain.shape(0)!=material.count() || strain.shape(1)!=material.components())
        throw py::value_error("Expected strain [material points, components]");
      auto response=material.update(values(strain));
      return py::make_tuple(array(response.stress,{material.count(),material.components()}),
                           array(response.tangent,{material.count(),material.components(),material.components()}));
    })
    .def("commit",&PythonMaterialBatch::commit)
    .def("cancel",&PythonMaterialBatch::cancel)
    .def("history_vjp",[](PythonMaterialBatch& material,const Input& history,
                           const Input& strain,const Input& adjoint,double step) {
      if (history.ndim()!=3 || history.shape(1)!=material.count() || history.shape(2)!=material.components())
        throw py::value_error("Expected history [steps, material points, components]");
      auto result=material.historyVjp(values(history),history.shape(0),values(strain),values(adjoint),step);
      return array(result,{history.shape(0),material.count(),material.components()});
    },py::arg("history"),py::arg("strain"),py::arg("adjoint"),py::arg("step")=1.e-7);
  py::class_<PythonMaterialFactory,std::shared_ptr<PythonMaterialFactory>>(module,"MaterialFactory")
    .def(py::init<const std::string&>(),py::arg("properties"))
    .def("__call__",[](const PythonMaterialFactory& factory,int count){return factory.create(count);})
    .def_static("types",&PythonMaterialFactory::types);
}

#ifdef FE2_PYTHON_EXTENSION
PYBIND11_MODULE(_fe2_material_native,module)
{
  // The public Python wrapper ensures this library is loaded on the main
  // thread: Jem's static initializers allocate before this function runs.
  // Bootstrap the executor before callbacks from temporary Python workers.
  PythonMaterialFactory::types();
  bindMaterials(module);
}
#else
PYBIND11_EMBEDDED_MODULE(_fe2_material,module) {bindMaterials(module);}
#endif
