#ifndef PYTHON_PRNN_MATERIAL_H
#define PYTHON_PRNN_MATERIAL_H

#include "Material.h"
#include <memory>

// Thin online driver. The Python model owns its architecture and material
// context; Jive owns the accepted/rejected increment lifecycle.
class PRNNMaterial : public Material
{
 public:
  JEM_DECLARE_CLASS ( PRNNMaterial, Material );
  static const char* TYPE_NAME;
  PRNNMaterial ( const String& name, const Properties& conf,
                       const Properties& props, const Properties& globdat );
  void configure ( const Properties& props, const Properties& globdat ) override;
  void getConfig ( const Properties& conf, const Properties& globdat ) const override;
  void allocPoints ( idx_t count ) override;
  void commit () override;
  void cancel () override;
  idx_t rank () const noexcept override;
  ProblemType getProblemType () const noexcept override;
  void update ( const Vector& stress, const Matrix& stiff,
                const Vector& strain, idx_t ip ) const override;
  void update ( const Vector& stress, const Vector& strain, idx_t ip ) const override;
  void elasticUpdate ( const Vector& stress, const Matrix& stiff,
                       const Vector& strain ) const override;
  Vector elasticUpdate ( const Vector& strain ) const override;
  Matrix getStiffMat () const override;
  Tuple<double,6> fill3DStrain ( const Vector& strain ) const override;
  Tuple<double,6> fill3DStress ( const Vector& stress ) const override;
  static Ref<Material> makeNew ( const String& name, const Properties& conf,
                                const Properties& props, const Properties& globdat );
  static void declare ();
 protected:
  ~PRNNMaterial () override;
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
#endif
