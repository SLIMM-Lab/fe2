
#include <jem/base/Once.h>


#include <jive/fem/InputModule.h>
#include <jive/app/ModuleFactory.h>


#include "declare.h"


#include "GmshInputModule.h"
#include "GroupInputModule.h"
#include "PBCGroupInputModule.h"
#include "MicroPBCSolverModule.h"


#include "SolidModel.h"
#include "DirichletModel.h"
#include "LoadDispModel.h"


#include "IsotropicMaterial.h"
#include "MelroMaterial.h"
#include "J2Material.h"
#include "FE2Material.h"
#ifdef FE2_WITH_LIBTORCH
#include "PRNNMaterial.h"
#endif


//=======================================================================
//   functions for modules
//=======================================================================


//-----------------------------------------------------------------------
//    utility functions declareModules
//-----------------------------------------------------------------------


void declareModules_

  ()

{
  using jive::app::ModuleFactory;
  using jive::fem::InputModule;


  GmshInputModule::declare      ();
  PBCGroupInputModule::declare  ();
  MicroPBCSolverModule::declare ();

  ModuleFactory::declare ( "Input", & InputModule::makeNew );
}


void declareModules

  ()

{
  static jem::Once once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, declareModules_ );
}


//=======================================================================
//   functions for models
//=======================================================================


//-----------------------------------------------------------------------
//    utility functions declareModels
//-----------------------------------------------------------------------


void declareModels_

  ()

{
  SolidModel::declare     ();
  DirichletModel::declare ();
  LoadDispModel::declare  ();
}


void declareModels 

  ()

{
  static jem::Once once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, declareModels_ );
}


//=======================================================================
//   functions for materials
//=======================================================================


//-----------------------------------------------------------------------
//    utility function: declareMaterials 
//-----------------------------------------------------------------------


void declareMaterials_ 

  ()

{
  IsotropicMaterial::declare ();
  MelroMaterial::declare     ();
  J2Material::declare        ();
  FE2Material::declare       ();
#ifdef FE2_WITH_LIBTORCH
  PRNNMaterial::declare      ();
#endif
}


void declareMaterials

  ()

{
  static jem::Once once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, declareMaterials_ );
}
