#ifndef PYTHON_MATERIAL_BRIDGE_H
#define PYTHON_MATERIAL_BRIDGE_H

#include "Material.h"
#include <memory>
#include <string>
#include <vector>

// Plain C++ data across the Python boundary; Jem objects remain on a
// Jem-managed executor thread, including when called by PyTorch workers.
struct MaterialResponse
{
  std::vector<double> stress, tangent;
};

class PythonMaterialFactory;

class PythonMaterialBatch
{
 public:
  ~PythonMaterialBatch ();
  idx_t count () const noexcept;
  idx_t components () const noexcept;
  MaterialResponse update ( const std::vector<double>& strain );
  void commit ();
  void cancel ();
  std::vector<double> historyVjp ( const std::vector<double>& history,
                                  idx_t steps, const std::vector<double>& strain,
                                  const std::vector<double>& adjoint, double step );
 private:
  friend class PythonMaterialFactory;
  struct State;
  explicit PythonMaterialBatch ( State* state, idx_t count, idx_t components );
  State* state_;
  idx_t count_, components_;
};

class PythonMaterialFactory
{
 public:
  explicit PythonMaterialFactory ( const std::string& properties );
  PythonMaterialFactory ( const Properties& materialProps, const Properties& globdat );
  ~PythonMaterialFactory ();
  std::shared_ptr<PythonMaterialBatch> create ( idx_t count ) const;
  static std::vector<std::string> types ();
 private:
  struct Specification;
  std::shared_ptr<Specification> spec_;
};

#endif
