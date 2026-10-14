
#include <jive/app/Application.h>
#include <jive/app/ChainModule.h>
#include <jive/app/ControlModule.h>
#include <jive/app/declare.h>
#include <jive/app/InfoModule.h>
#include <jive/app/ReportModule.h>
#include <jive/app/UserconfModule.h>

#include <jive/fem/InitModule.h>
#include <jive/fem/ShapeModule.h>
#include <jive/fem/declare.h>
#include <jive/geom/declare.h>
#include <jive/gl/declare.h>

#include <jive/implict/declare.h>

#include <jive/model/declare.h>

#include "declare.h"


using jem::Ref;
using jem::newInstance;


using jive::app::Application;
using jive::app::ChainModule;
using jive::app::ControlModule;
using jive::app::InfoModule;
using jive::app::Module;
using jive::app::ReportModule;
using jive::app::UserconfModule;

using jive::fem::InitModule;
using jive::fem::ShapeModule;


//-----------------------------------------------------------------------
//   mainModule
//-----------------------------------------------------------------------


Ref<Module> mainModule 
  
  ()

{
  // Declare internal shapes, models, modules, and matrix builders, and 
  // so on. These functions essentially store points to construction 
  // functions that are called when Jive needs to create a shape, model 
  // or matrix builder of a particular type.
  
  jive::fem::declareMBuilders   ();
  jive::model::declareModels    ();
  jive::geom::declareIShapes    ();
  jive::geom::declareShapes     ();
  jive::implict::declareModules ();
  jive::app::declareModules     ();
#ifndef FE2_HEADLESS
  jive::gl::declareModules      ();
#endif

  // ... also, declare user-developed modules, models and materials.

  declareModules   ();
  declareModels    ();
  declareMaterials ();

  // Set up the module chain. ChainModule encapsulate a list of modules
  // that will be called by Jive in the order that they have been added 
  // to the chain.

  Ref<ChainModule> chain = newInstance<ChainModule> ();

  chain->pushBack ( newInstance<UserconfModule> ( "userInput" ) );
  chain->pushBack ( newInstance<ShapeModule>    ( "shape"     ) );
  chain->pushBack ( newInstance<InitModule>     ( "init"      ) );
  chain->pushBack ( newInstance<InfoModule>     ( "info"      ) );
  chain->pushBack ( newInstance<UserconfModule> ( "extra"     ) );

  // Wrap the entire chain in a ControlModule. It can be used to 
  // run a program in a interactive way, allowing the user to enter
  // commands through a terminal or a file. The user can specify a
  // condition that controls when the program is terminated.
  
  Ref<ControlModule> ctrl = newInstance<ControlModule> 
                                                    ( "control", chain );

  // Finally, wrap aroung the ctrl module. The ReportModule keeps track
  // of all properties that have been used by its child modules, and 
  // prints an overview of all properties at the end of its init member
  // function (precisely, the conf properties set afer all modules and 
  // models that have been initialized).

  return newInstance<ReportModule> ( "report", ctrl );
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ( int argc, char** argv )

{
  return Application::exec ( argc, argv, & mainModule );
}
