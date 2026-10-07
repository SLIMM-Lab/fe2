
/*
 *  Copyright (C) 2018 TU Delft. All rights reserved.
 *  
 *  This class implements the material model based on
 *  classic Von Mises J2-theory.
 *  This class is a child material member of DamTLSMaterial
 *  classes.
 *
 *  NB.: This class only works on Plane Stress assumptions.
 *       However, the architecture for Plane Strain and 3D
 *       is already defined. According to [1], it is possible
 *       to write the governing equations only related to 
 *       plastic multiplier.
 *  
 *  Author: Luiz A. T. Mororo, l.a.taumaturgomororo@tudelft.nl
 *  Date:   Nov 2018
 *
 *  [1] E.A. de Souza Neto, D. Peric, D.R.J. Owen. Computational
 *      Methods for Plasticity: Theory and Applications. Jhon Wiley
 *      and Sons, 2008.
 *  
 *  Changelog:
 *
 *  Modified: Iuri Rocha, i.rocha@tudelft.nl
 *  Date: Feb 2022
 *
 *  Adapted to ML use. Switched to linear hardening
 *
 *  Modified: Luiz A. T. Mororo, luiz.mororo@ifce.edu.br
 *  Date: Sep 2025
 *
 *  Clean-up code, remove TLS and ML functionalities, and refactored 
 *  member functions for better performance, storage scheme, and to meet 
 *  new Jem/Jive (version 3.0) data types, for instance, `idx_t'. 
 */


#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/System.h>
#include <jem/base/Float.h>
#include <jem/io/PrintWriter.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/func/UserFunc.h>


#include <jive/util/FuncUtils.h>


#include "J2Material.h"
#include "MaterialFactory.h"


JEM_DEFINE_CLASS ( J2Material );


using namespace jem::literals;


using jem::numeric::matmul;


//=======================================================================
//   YieldFunc_ class
//=======================================================================


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


YieldFunc_::YieldFunc_

  ( double               young,
    double               poisson,
    const Ref<Function>  sigY )

  : sigmaY_ ( sigY ), rmTol_(0.), maxIter_(0)

{
  E_       = young;
  nu_      = poisson;
  G_       = young / 2. / ( 1. + poisson );
  K_       = young / 3. / ( 1. - 2. * poisson );

  rankDep_ = 3;
}


YieldFunc_::~YieldFunc_

  ()

{}


//-----------------------------------------------------------------------
//   setRmSettings
//-----------------------------------------------------------------------


void YieldFunc_::setRmSettings 

  ( double  rmTolerance, 
    idx_t   rmMaxIter )

{
  rmTol_   = rmTolerance;
  maxIter_ = rmMaxIter;
}


//-----------------------------------------------------------------------
//   setElasticStiff
//-----------------------------------------------------------------------


void YieldFunc_::setElasticStiff

  ( const Matrix& mat ) 

{
  const idx_t nr = mat.size(0);
  const idx_t nc = mat.size(1);

  JEM_ASSERT ( (nr == 3_idx && nc == 3_idx) || 
               (nr == 6_idx && nc == 6_idx) );

  rankDep_ = nr;

  De_.resize ( nr, nc );

  for ( idx_t i = 0; i < nr; ++i )
  {
    for ( idx_t j = 0; j < nc; ++j )
    {
      De_(i,j) = mat(i,j);
    }
  }
}


//-----------------------------------------------------------------------
//   findRoot 
//-----------------------------------------------------------------------


double YieldFunc_::findRoot

  ( double  dgam0 ) 

{
  double dgam     = dgam0;
  double oldddgam = -1.;
  double oldcrit  = evalTrial_ ();

  for ( idx_t irm = 0; irm < maxIter_; )
  {
    // Evaluate the yield functions.

    double crit = eval_ ( dgam );

    // Compute the derivative of yield function with respect to dgam.
    // (residual derivative)

    double df   = evalDer_ ( dgam );

    if ( jem::numeric::abs( crit ) < rmTol_ )
    {
      break;
    }
    if ( ++irm == maxIter_ )
    {
      throw Exception ( JEM_FUNC, "No convergence" );
    }
    if ( jem::Float::isNaN( df ) )
    {
      throw Exception ( JEM_FUNC, "nan" );
    }

    // Corrector.

    double ddgam = crit / df;

    // Divergence detection.

    if ( ddgam * oldddgam < 0. )
    {
      if ( oldcrit * crit < 0. && 
      
           jem::numeric::abs( ddgam ) > jem::numeric::abs( oldddgam ) )
      {
        // there might be an inflection point around the root
        // use linear interpolation rather than linearization

        ddgam = -oldddgam * crit / ( crit - oldcrit );
      }
    }

    // Set a new guess for dgam.

    dgam    -= ddgam;

    oldddgam = ddgam;
    oldcrit  = crit;
  }

  if ( dgam < -1.e-12 )
  {
    throw Exception ( JEM_FUNC, "Negative dgam" );
  }

  return dgam;
}


//-----------------------------------------------------------------------
//   findBounds 
//-----------------------------------------------------------------------


void YieldFunc_::findBounds

  ( double&  gmin,
    double&  gmax,
    double&  fmin,
    double&  fmax ) 

{
  gmin = 0.;
  gmax = -1.;
  fmax = -1.;

  fmin = evalTrial_();

  JEM_ASSERT ( fmin > 0. );

  double dgam = 1.e-16;

  while ( gmax < 0. )
  {
    double crit = eval_ ( dgam );

    if ( crit > 0. )
    {
      gmin = dgam;
      fmin = crit;
    }
    else
    {
      gmax = dgam;
      fmax = crit;
    }
    dgam *= 10.;
  }
}


//-----------------------------------------------------------------------
//   findRoot 
//-----------------------------------------------------------------------


double YieldFunc_::findRoot

  ( double  gmin,
    double  gmax,
    double  fmin,
    double  fmax )

{
  double dgam;

  try 
  {
    double dgam0 = estimateRoot_ ( gmin, gmax, fmin, fmax );

    dgam         = findRoot      ( dgam0 );
  }
  catch ( const Exception& ex )
  {
    improveBounds_  ( gmin, gmax, fmin, fmax );

    dgam = findRoot ( gmin, gmax, fmin, fmax );
  }

  return dgam;
}


//-----------------------------------------------------------------------
//   deviatoric_
//-----------------------------------------------------------------------


Vec6 YieldFunc_::deviatoric_

  ( const Vec6&   full,
    const double  p ) const

{
  Vec6 ret = full;

  ret[0] -= p;
  ret[1] -= p;
  ret[2] -= p;

  return ret;
}


//-----------------------------------------------------------------------
//   estimateRoot_ 
//-----------------------------------------------------------------------


double YieldFunc_::estimateRoot_

  ( double  gmin,
    double  gmax,
    double  fmin,
    double  fmax ) const

{
  // Linear interpolation.

  // fmin  o___ 
  //           ---___
  // phi=0 |----------o-----------|------
  //                      ---___
  // fmax  -                    --o
  //     gmin        ret        gmax

  return gmin + fmin * (gmax-gmin) / (fmin-fmax);
}


//-----------------------------------------------------------------------
//   improveBounds_ 
//-----------------------------------------------------------------------


void YieldFunc_::improveBounds_

  ( double&  gmin,
    double&  gmax,
    double&  fmin,
    double&  fmax )

{
  idx_t  np   = 10;
  double dg   = ( gmax - gmin ) / double ( np );
  double dgam = gmin;

  while ( dgam < gmax )
  {
    dgam += dg;

    double crit = eval_ ( dgam );

    if ( crit > 0. )
    { 
      gmin = dgam;
      fmin = crit;
    }
    else
    {
      gmax = dgam;
      fmax = crit;

      break;
    }
  }
}


//=======================================================================
//   YieldFunc3D_ class
//=======================================================================


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


YieldFunc3D_::YieldFunc3D_

  ( double               young,
    double               poisson,
    const Ref<Function>  yield    )

  : YieldFunc_ ( young, poisson, yield )

{
  sigDtr_ = 0.;
  ptrial_ = 0.;
}

YieldFunc3D_::~YieldFunc3D_

  ()

{}


//-----------------------------------------------------------------------
//   isLinear 
//-----------------------------------------------------------------------


bool YieldFunc3D_::isLinear

  ( const Vec6&   sigtr,
    double        epspeq0 )

{
  epspeq0_ = epspeq0; 

  StressInvariants inv  ( sigtr );

  // Hydrostatic stress tensor.

  ptrial_ = inv.getFirstInvariant () / 3.;

  // Deviatoric stress tensor.

  sigDtr_ = deviatoric_ ( sigtr, ptrial_ );

  // J2 term => J2 = 1/2 * (s:s).

  double J2 = inv.getJ2 ();

  ksitr_    = std::sqrt ( 3. * J2 );

  return evalTrial_() < rmTol_; 
}


//-----------------------------------------------------------------------
//   compSigUpdated 
//-----------------------------------------------------------------------


void YieldFunc3D_::compSigUpdated

  ( Vec6&        sig, 
    double       dgam, 
    const Vec6&  sigtr ) 

{
  const double frac = 1. - dgam * 3. * G_ / ksitr_;

  sigD_ = frac * sigDtr_;

  sig   = frac * sigDtr_; 

  sig[0] += ptrial_;
  sig[1] += ptrial_;
  sig[2] += ptrial_;
}


//-----------------------------------------------------------------------
//   compPlastStrInc 
//-----------------------------------------------------------------------


void YieldFunc3D_::compPlastStrInc

  ( Vec6&   depsp, 
    double  dgam )

{
  // Before calling getPlastStrInc(), it must call compSigUpdated() first
  // to compute sigD_ member.

  const double normS = normSig_ ( sigD_ );

  depsp = dgam * std::sqrt ( 1.5 ) * sigD_ / normS;

  depsp[3] *= 2.;
  depsp[4] *= 2.;
  depsp[5] *= 2.;
}


//-----------------------------------------------------------------------
//   compEquPlastInc 
//-----------------------------------------------------------------------


double YieldFunc3D_::compEquPlastInc

  ( double  dgam ) 

{
  return dgam;
}


//-----------------------------------------------------------------------
//   compTangStiff 
//-----------------------------------------------------------------------


Matrix YieldFunc3D_::compTangStiff
  
  ( const double  dgam )

{
  // Unit plastic flow vector related to trial stress.

  double normStr = normSig_ ( sigDtr_ ); 

  Vec6   N       = sigDtr_ / normStr; 

  // Hardening slope.

  double H = sigmaY_->deriv ( epspeq_ );

  // Compute elastoplastic tangent matrix. 

  double fac1 = 2. * G_ * ( 1. - dgam * 3. * G_ / ksitr_ );

  double fac2 = 6. * G_ * G_ * ( dgam / ksitr_ - 1. / ( 3.*G_ + H ) );

  Tuple<double,6,6> NxN;
  matmul ( NxN, N, N );

  Tuple<double,6,6> IxI;
  IxI = 0.;
  for ( idx_t i = 0; i < 3; ++i) 
  {
    for ( idx_t j = 0; j < 3; ++j )
    {
      IxI(i,j) = 1.;
    }
  }

  Tuple<double,6,6> Id;
  Id = getIdMatrix_ ();

  Tuple<double,6,6> dmat6;
  dmat6 = fac1 * Id + fac2 * NxN + K_ * IxI;

  Matrix Dep ( 6, 6 );
  for ( idx_t i = 0; i < 6; ++i )
  {
    for ( idx_t j = 0; j < 6; ++j )
    {
      Dep(i,j) = dmat6(i,j);
    }
  }

  if ( rankDep_ == 3 )
  {
    Matrix Dep33 ( 3, 3 );
    m66_to_m33_ ( Dep33, Dep );

    return Dep33;
  }
  else
  {
    return Dep;
  }
}


//-----------------------------------------------------------------------
//   evalTrial_
//-----------------------------------------------------------------------


double YieldFunc3D_::evalTrial_

  () const

{
  double sigY = sigmaY_->eval ( epspeq0_ );

  return ksitr_ - sigY;
}


//-----------------------------------------------------------------------
//   eval_
//-----------------------------------------------------------------------


double YieldFunc3D_::eval_

  ( double  dgam )

{
  epspeq_     = epspeq0_ + dgam; 

  double sigY = sigmaY_->eval ( epspeq_ );

  double fac  = evalKsi_      ( dgam );

  return fac - sigY;  
}


//-----------------------------------------------------------------------
//   evalDer_
//-----------------------------------------------------------------------


double YieldFunc3D_::evalDer_

  ( double  dgam ) 

{
  // Hardening slope.

  double H = sigmaY_->deriv ( epspeq_ );

  // Residual derivative.

  return -3. * G_ - H ;
}


//-----------------------------------------------------------------------
//   evalKsi_
//-----------------------------------------------------------------------


double YieldFunc3D_::evalKsi_

  ( double  dgam ) 

{
  return ksitr_ - 3. * G_ * dgam; 
}


//-----------------------------------------------------------------------
//   evalDerKsi_
//-----------------------------------------------------------------------


double YieldFunc3D_::evalDerKsi_

  ( double  dgam ) 

{
  return ksitr_ - 3. * G_;
}


//-----------------------------------------------------------------------
//   getIdMatrix_ 
//-----------------------------------------------------------------------


Tuple<double,6,6> YieldFunc3D_::getIdMatrix_

  () 

{
  Tuple<double,6,6> Id;

  Id = 0.;

  Id(0,0) =  2./3.; Id(0,1) = -1./3.; Id(0,2) = -1./3.;

  Id(1,0) = -1./3.; Id(1,1) =  2./3.; Id(1,2) = -1./3.; 

  Id(2,0) = -1./3.; Id(2,1) = -1./3.; Id(2,2) =  2./3.;

  Id(3,3) =  1./2.; 
  Id(4,4) =  1./2.;
  Id(5,5) =  1./2.;

  return Id;
}


//-----------------------------------------------------------------------
//   normSig_ 
//-----------------------------------------------------------------------


double YieldFunc3D_::normSig_

  ( const Vec6&  sig ) const 

{
  // Compute the norm of 'stress' tensors.

  return std::sqrt ( sig[0]*sig[0] + sig[1]*sig[1] + sig[2]*sig[2] +

                2.*sig[3]*sig[3] + 2.*sig[4]*sig[4] + 2.*sig[5]*sig[5] );
}


//-----------------------------------------------------------------------
//   m66_to_m33_ 
//-----------------------------------------------------------------------


void YieldFunc3D_::m66_to_m33_

  ( const Matrix&  m33,
    const Matrix&  m66 ) const

{
  // Selecting [xx, yy, xy] components from [xx, yy, zz, xy, yz, xz]
  // This member is designed to P.Strain. 
 
  for ( idx_t i = 0; i < 2; ++i )
  {
    m33(i,2) = m66(i,3);
    m33(2,i) = m66(3,i);
 
    for ( idx_t j = 0; j < 2; ++j )
    {
      m33(i,j) = m66(i,j);
    }
  }
  
  m33(2,2) = m66(3,3);
}


//=======================================================================
//   YieldFuncPS_ class
//=======================================================================


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


YieldFuncPS_::YieldFuncPS_

  ( double               young,
    double               poisson,
    const Ref<Function>  yield    )

  : YieldFunc_ ( young, poisson, yield )

{
  P_ = 0.;

  P_(0,0) = P_(1,1) =  2./3.;
  P_(0,1) = P_(1,0) = -1./3.;
  P_(2,2) = 2.;

  stress_ = 0.;
}


YieldFuncPS_::~YieldFuncPS_

  ()

{}


//-----------------------------------------------------------------------
//   isLinear 
//-----------------------------------------------------------------------


bool YieldFuncPS_::isLinear

  ( const Vec6&  sigtr,
    double       epspeq0 )

{
  epspeq0_ = epspeq0; 

  a1tr_  = ( sigtr[0] + sigtr[1] ) * ( sigtr[0] + sigtr[1] );
  a2tr_  = ( sigtr[1] - sigtr[0] ) * ( sigtr[1] - sigtr[0] );
  a3tr_  = sigtr[3] * sigtr[3];

  ksitr_ = a1tr_/6. + 0.5*a2tr_ + 2.*a3tr_;

  return evalTrial_() < rmTol_; 
}


//-----------------------------------------------------------------------
//   compSigUpdated 
//-----------------------------------------------------------------------


void YieldFuncPS_::compSigUpdated

  ( Vec6&        sig, 
    double       dgam, 
    const Vec6&  sigtr ) 

{
  Vec3 sigtr3;

  t6_to_t3_ ( sigtr3, sigtr );

  // Compute A matrix.

  Tuple<double,3,3> A ( getAMatrix_( dgam ) );

  // Compute the updated stress.

  stress_ = matmul ( A, sigtr3 );

  t3_to_t6_     ( sig, stress_ );
}


//-----------------------------------------------------------------------
//   compPlastStrInc 
//-----------------------------------------------------------------------


void YieldFuncPS_::compPlastStrInc

  ( Vec6&   depsp, 
    double  dgam )

{
  // Before calling getPlastStrInc(), it must call compSigUpdated() first
  // to compute stress_ member.

  Vec3 depsp3;
  
  depsp3 = dgam * matmul ( P_, stress_ );

  ep3_to_ep6_ ( depsp, depsp3 );

  //depsp[3] *= 2.;
}


//-----------------------------------------------------------------------
//   compEquPlastInc 
//-----------------------------------------------------------------------


double YieldFuncPS_::compEquPlastInc

  ( double  dgam ) 

{
  double fac = 2. * evalKsi_( dgam ) / 3.;  

  return dgam * std::sqrt( fac );
}


//-----------------------------------------------------------------------
//   compTangStiff 
//-----------------------------------------------------------------------


Matrix YieldFuncPS_::compTangStiff
  
  ( double  dgam )

{
  // Hardening slope.

  double H = sigmaY_->deriv ( epspeq_ );

  // Compute ksi for stress and stiffness.

  double k = dot ( matmul( P_, stress_ ), stress_ );

  // Matrix E.

  Tuple<double,3,3> E;
  
  E = getEMatrix_ ( dgam );

  // Vector n.

  Vec3 n = matmul ( E, matmul( P_, stress_ ) );

  // Alpha.

  double fac   = dot ( matmul( stress_, P_ ), n );
  fac         += 2. * k * H / ( 3. - 2. * H * dgam );

  double alpha = 1. / fac;

  // Assemble the elastoplastic tangent matrix.

  Tuple<double,3,3> dmat3;
  
  dmat3 = E - alpha * matmul ( n, n.transpose() );

  Matrix D3 ( 3, 3 );

  t33_to_m33_ ( D3, dmat3 );

  return D3;
}


//-----------------------------------------------------------------------
//   evalTrial_
//-----------------------------------------------------------------------


double YieldFuncPS_::evalTrial_

  () const

{
  double sigY = sigmaY_->eval ( epspeq0_ ) ;

  return 0.5*ksitr_ - sigY*sigY/3.;
}


//-----------------------------------------------------------------------
//   eval_
//-----------------------------------------------------------------------


double YieldFuncPS_::eval_

  ( double  dgam )

{
  double ksi  = evalKsi_ ( dgam );

  epspeq_     = epspeq0_ + dgam * std::sqrt ( 2. * ksi / 3. ); 

  double sigY = sigmaY_->eval ( epspeq_ );

  return 0.5*ksi - sigY*sigY/3.;  
}


//-----------------------------------------------------------------------
//   evalDer_
//-----------------------------------------------------------------------


double YieldFuncPS_::evalDer_

  ( const double  dgam ) 

{
  // Compute ksi. 

  double ksi = evalKsi_ ( dgam );

  // ksiDer.

  double ksiDer = evalDerKsi_ ( dgam );

  // sigY and H

  double sigY = sigmaY_->eval  ( epspeq_ );
  double H    = sigmaY_->deriv ( epspeq_ );

  // Hbar.

  double sqrtKsi = std::sqrt   ( ksi );

  double Hbar = 2.*sigY * H * sqrt(2./3.) * 

               ( sqrtKsi + dgam*ksiDer/2./sqrtKsi );

  return 0.5*ksiDer - Hbar/3.;
}


//-----------------------------------------------------------------------
//   evalKsi_
//-----------------------------------------------------------------------


double YieldFuncPS_::evalKsi_

  ( double  dgam ) 

{
  double f1  = 1. + E_ * dgam / 3. / ( 1. - nu_ );
  double f2  = 1. + 2. * G_ * dgam;

  double f12 = f1 * f1 * 6.;
  double f22 = f2 * f2;

  return a1tr_ / f12 + ( 0.5*a2tr_ + 2.*a3tr_ ) / f22;
}


//-----------------------------------------------------------------------
//   evalDerKsi_
//-----------------------------------------------------------------------


double YieldFuncPS_::evalDerKsi_

  ( double  dgam ) 

{
  double fac = E_ / ( 1. - nu_ );

  double f1  = 1. + fac * dgam / 3.;
  double f2  = 1. + 2.  * G_ * dgam;

  double f13 = f1 * f1 * f1 * 9.;
  double f23 = f2 * f2 * f2;

  return - a1tr_ * fac / f13 - 2. * G_ * ( a2tr_ + 4.*a3tr_ ) / f23; 
}


//-----------------------------------------------------------------------
//   getAMatrix_ 
//-----------------------------------------------------------------------


Tuple<double,3,3> YieldFuncPS_::getAMatrix_

  ( double  dgam ) const 

{
  double A11 = 3. * ( 1. - nu_ ) / ( 3. * ( 1. - nu_ ) + E_ * dgam );
  double A22 = 1. / ( 1 + 2. * G_ * dgam );
  double A33 = A22;

  Tuple<double,3,3> tmp;

  tmp = 0.;

  tmp(0,0) = .5 * ( A11 + A22 ); tmp(0,1) = .5 * ( A11 - A22 );
  tmp(1,0) = .5 * ( A11 - A22 ); tmp(1,1) = .5 * ( A11 + A22 );  
  tmp(2,2) = A33; 

  return tmp;
}


//-----------------------------------------------------------------------
//   getEMatrix_ 
//-----------------------------------------------------------------------


Tuple<double,3,3> YieldFuncPS_::getEMatrix_

  ( double  dgam ) const 

{
  double E11 = 3. * E_ / ( 3. * ( 1. - nu_ ) + E_ * dgam );
  double E22 = 2. * G_ / ( 1 + 2. * G_ * dgam );
  double E33 = .5 * E22; 

  Tuple<double,3,3> tmp;

  tmp = 0.;

  tmp(0,0) = .5 * ( E11 + E22 ); tmp(0,1) = .5 * ( E11 - E22 );
  tmp(1,0) = .5 * ( E11 - E22 ); tmp(1,1) = .5 * ( E11 + E22 );  
  tmp(2,2) = E33; 

  return tmp;
}


//-----------------------------------------------------------------------
//   t33_to_m33_
//-----------------------------------------------------------------------


void YieldFuncPS_::t33_to_m33_
 
  ( const Matrix&             m,
    const Tuple<double,3,3>&  t ) const

{
  for ( idx_t i = 0; i < 3; ++i )
  {
    for ( idx_t j = 0; j < 3; ++j )
    {
      m(i,j) = t(i,j);
    }
  }
}


//=======================================================================
//   class J2Material::Hist_
//=======================================================================


//-----------------------------------------------------------------------
//   Hist_ constructor
//-----------------------------------------------------------------------


J2Material::Hist_::Hist_ 
   
  ()
     
{
  epsp        = 0.;
  epspeq      = 0.;
  dissipation = 0.;
  loading     = false;
}

/*
void J2Material::Hist_::toVector

  ( Vector& vec )

{
  vec.resize ( 7 );

  vec = 0.0;

  vec[0] = epsp[0];
  vec[1] = epsp[1];
  vec[2] = epsp[2];
  vec[3] = epsp[3];
  vec[4] = epsp[4];
  vec[5] = epsp[5];
  vec[6] = epspeq;
}
*/


//=======================================================================
//   class J2Material 
//=======================================================================


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  J2Material::TYPE_NAME    = "J2";

const char*  J2Material::RM_TOLERANCE = "rmTolerance";
const char*  J2Material::RM_MAX_ITER  = "rmMaxIter";
const char*  J2Material::YIELD_PROP   = "yield"; 


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


J2Material::J2Material

  ( const String&      name, 
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

    : Super ( name, globdat, 3_idx )

{
  // Get material Properties.

  Properties myProps = props.getProps ( myName_ );
  Properties myConf  = conf.makeProps ( myName_ );

  // Get "dim" (or "rank" of Melro material) parameter.

  j2Rank_ = -1_idx;

  myProps.get ( j2Rank_  , RANK_PROP, 2_idx, 3_idx ); 
  myConf .set ( RANK_PROP, j2Rank_ );

  // Default members.

  rmTolerance_ = 1.e-10;
  rmMaxIter_   = 10_idx;

  v61_.resize ( 6 );
  v62_.resize ( 6 );
}


J2Material::~J2Material

  ()

{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void J2Material::configure 

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::newInstance;


  // First, let Super material handle its own configure action.

  Super::configure ( props, globdat );

  // Then, J2Material deals with its onw configure member function.

  Properties myProps = props.getProps ( myName_ );

  myProps.find   ( rmTolerance_, RM_TOLERANCE     );
  myProps.find   ( rmMaxIter_  , RM_MAX_ITER      );

  if      ( j2Rank_ == 2_idx )
  {
    myProps.get ( stateString_, STATE_PROP );

    if      ( stateString_ == "PlaneStrain" )
    {
      state_ = ProblemType::PlaneStrain;
    }
    else if ( stateString_ == "PlaneStress" )
    {
      state_ = ProblemType::PlaneStress;
    }
    else
    {
      throw jem::IllegalInputException ( 
        CLASS_NAME, 
        "it only works on P.Stress and P.Strain!" 
      );
    }

    // Compute the 3x3 matrix for 2D problems.

    comp2DStiffMat_ ();
  }
  else if ( j2Rank_ == 3_idx )
  {
    state_ = ProblemType::Solid;
     
    stateString_ = "Solid";
  }
  else
  {
    throw jem::IllegalInputException ( 
      CLASS_NAME, 
      "this material only works on P.Stress, P.Strain and 3D!" 
    );
  }

  // Set YieldFunc_ member.

  Ref<Function> sigmaY = makeFunc_ ( YIELD_PROP, myProps, globdat );

  if ( j2Rank_ == 2_idx )
  {
    if      ( state_ == ProblemType::PlaneStrain )
    { 
      y_ = newInstance<YieldFunc3D_> ( young(), poisson(), sigmaY );

      y_-> setRmSettings             ( rmTolerance_, rmMaxIter_ );
      y_-> setElasticStiff           ( stiff2D_ );
    }
    else if ( state_ == ProblemType::PlaneStress )
    {
      y_ = newInstance<YieldFuncPS_> ( young(), poisson(), sigmaY );

      y_-> setRmSettings             ( rmTolerance_, rmMaxIter_ );
      y_-> setElasticStiff           ( stiff2D_ );
    }
    else
    {
      throw jem::IllegalInputException ( 
        CLASS_NAME, 
        "it only works on P.Stress and P.Strain!" 
      );
    }
  }
  else if ( j2Rank_ == 3_idx )
  {
    y_ = newInstance<YieldFunc3D_> ( young(), poisson(), sigmaY );

    y_-> setRmSettings             ( rmTolerance_, rmMaxIter_ );
    y_-> setElasticStiff           ( Super::getStiffMat() );
  }
  else
  {
    throw jem::IllegalInputException ( 
      CLASS_NAME, 
      "this material only works on P.Stress, P.Strain and 3D!" 
    );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void J2Material::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const 

{
  using jive::util::FuncUtils;


  Super::getConfig ( conf, globdat );


  Properties myConf = conf.makeProps ( myName_ );

  myConf.set ( RM_TOLERANCE, rmTolerance_ );
  myConf.set ( RM_MAX_ITER , rmMaxIter_   );

  FuncUtils::getConfig ( myConf, y_->getSigmaYFunc(), YIELD_PROP );

  if ( j2Rank_ == 2 )
  {
    // Note that state is not set, if Super has rank 3.

    myConf.set ( STATE_PROP, stateString_ );
  }
}


//-----------------------------------------------------------------------
//   commit 
//-----------------------------------------------------------------------


void J2Material::commit 

  () 

{
  newHist_.swap ( preHist_ );

  latestHist_ = & preHist_;
}


//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------

void J2Material::cancel ()
{
  for ( idx_t i = 0; i < preHist_.size(); ++i )
  {
    newHist_[i] = preHist_[i];
  }
  latestHist_ = & preHist_;
}


//-----------------------------------------------------------------------
//   allocPoints
//-----------------------------------------------------------------------


void J2Material::allocPoints

  ( idx_t  count )

{
  for ( idx_t i = 0; i < count; i++ )
  {
    preHist_.pushBack ( Hist_() );
    newHist_.pushBack ( Hist_() );
  }

  latestHist_ = & preHist_;
}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t J2Material::rank

  () const noexcept

{
  return j2Rank_;
}


//-----------------------------------------------------------------------
//   getProblemType
//-----------------------------------------------------------------------


ProblemType J2Material::getProblemType 

  () const noexcept

{
  return state_;
}

//-----------------------------------------------------------------------
//   elasticUpdate 
//-----------------------------------------------------------------------


void J2Material::elasticUpdate

  ( const Vector&  stress,
    const Matrix&  stiff,
    const Vector&  strain ) const

{
  if ( j2Rank_ == 3_idx )
  {
    Super::elasticUpdate ( stress, stiff, strain );
  }
  else
  {
    stiff = stiff2D_;

    matmul ( stress, stiff, strain );
  }
}


//-----------------------------------------------------------------------
//   elasticUpdate 
//-----------------------------------------------------------------------


Vector J2Material::elasticUpdate

  ( const Vector&  strain ) const

{
  if ( j2Rank_ == 3_idx )
  {
    return Super::elasticUpdate ( strain );
  }
  else
  {
    return matmul ( stiff2D_, strain );
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void J2Material::update

  ( const Vector&  stress,
    const Matrix&  stiff,
    const Vector&  strain,
    const idx_t    ipoint ) const

{
  // Get equivalent plastic strain (previous).

  double epspeq0 = preHist_[ipoint].epspeq;

  // Compute the trial elastic stress.

  Matrix stiff3D ( 6, 6 );
  stiff3D = Super::getStiffMat ();

  Vec6 epse6;
  epse6 = 0.0;

  if ( state_ == ProblemType::PlaneStress )
  {
    const double nu = poisson ();

    epse6[0] = strain[0] - preHist_[ipoint].epsp[0];
    epse6[1] = strain[1] - preHist_[ipoint].epsp[1];
    epse6[2] = -nu / (1.-nu) * (epse6[0] + epse6[1]);
    epse6[3] = strain[2] - preHist_[ipoint].epsp[3];
  }
  else
  {
    epse6 = fill3DStrain ( strain ) - preHist_[ipoint].epsp;
  }
 
  Vec6 sigtr = tmatmul_ ( stiff3D, epse6 );
   
  // Elastic predictor/return mapping scheme.

  Ref<Self> self  = const_cast<Self*> ( this ); // remove constness
                                                // updt. history-related
                                                // member variables

  Vec6   sig;
  double dgam;
  bool   loading = false;

  if ( y_->isLinear( sigtr, epspeq0 ) )
  {
    // Linear domain.

    sig = sigtr;

    stiff = y_->getElastStiff ();

    // Update history.

    self->newHist_[ipoint].epsp        = preHist_[ipoint].epsp;
    self->newHist_[ipoint].epspeq      = preHist_[ipoint].epspeq;
    self->newHist_[ipoint].dissipation = preHist_[ipoint].dissipation;
  }
  else
  {
    // Plastic flow takes place.

    try
    {
      // Classical return mapping scheme starting with dgam = 0.

      dgam = y_->findRoot ( 0. );
    }
    catch ( const Exception& ex ) 
    {
      handleException_ ( ex, strain, ipoint );

      // More robust return mapping scheme.

      double gmin, gmax, fmin, fmax;

      y_->findBounds      ( gmin, gmax, fmin, fmax );

      dgam = y_->findRoot ( gmin, gmax, fmin, fmax );
    }

    // Compute stress.

    y_->compSigUpdated ( sig, dgam, sigtr ); 

    // Compute plastic strain increment.

    Vec6 depsp;

    y_->compPlastStrInc ( depsp, dgam );

    // Compute elastoplastic tangent stiffness.

    stiff = y_->compTangStiff ( dgam );

    // Update history.

    double dG = dot ( sig, depsp );

    self->newHist_[ipoint].epsp        = preHist_[ipoint].epsp + depsp;
    self->newHist_[ipoint].epspeq      = y_->getEpspeq ();
    self->newHist_[ipoint].dissipation = preHist_[ipoint].dissipation + dG;

    loading = true; 
  }

  self->newHist_[ipoint].loading = loading;

  reduce3DVector_ ( stress, sig );

  self->latestHist_  = &( self->newHist_ );
}


void J2Material::update

  ( const Vector&  stress,
    const Vector&  strain,
    const idx_t    ipoint ) const

{
  // Compute elastic strain field.

  Vec6 eps    ( fill3DStrain( strain ) );  // total   strain

  Vec6 epsp = (*latestHist_)[ipoint].epsp; // plastic strain

  Vec6 epse = eps - epsp;                  // elastic strain

  // Get 3D elastic stiffness matrix.

  Matrix stiff3D ( 6, 6 );
  stiff3D = Super::getStiffMat ();

  // Compute stress field.

  Vec6 sig = tmatmul_ ( stiff3D, epse );
  reduce3DVector_     ( stress , sig  );
}


//-----------------------------------------------------------------------
//   getStiffMat 
//-----------------------------------------------------------------------


Matrix J2Material::getStiffMat 

  () const

{
  if ( j2Rank_ == 3_idx )
  {
    return Super::getStiffMat();
  }
  else
  {
    return stiff2D_;
  }
}


//-----------------------------------------------------------------------
//   makeNew 
//-----------------------------------------------------------------------

 
Ref<Material> J2Material::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return jem::newInstance<Self> ( name, conf, props, globdat ); 
}


//-----------------------------------------------------------------------
//   declare 
//-----------------------------------------------------------------------


void J2Material::declare

  ()

{
  MaterialFactory::declare ( TYPE_NAME,  & makeNew );
  MaterialFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   tmatmul_
//-----------------------------------------------------------------------


Vec6 J2Material::tmatmul_

  ( const Matrix&  mat,
    const Vec6&    t6 ) const

{
  t6_to_v6_ ( v61_, t6 );

  matmul    ( v62_, mat, v61_ );

  return v6_to_t6_ ( v62_ );
}


//-----------------------------------------------------------------------
//   makeFunc_
//-----------------------------------------------------------------------


Ref<Function> J2Material::makeFunc_ 

  ( const String&      name, 
    const Properties&  props,
    const Properties&  globdat ) const

{
  using jive::util::FuncUtils;


  String args = "x";
  props.find ( args, "args" );

  Ref<Function> func = FuncUtils::newFunc ( args, name, props, globdat );

  FuncUtils::resolve ( *func, globdat );

  return func;
}


//-----------------------------------------------------------------------
//   handleException_
//-----------------------------------------------------------------------


void J2Material::handleException_ 

  ( const Exception&  ex,
    const Vector&     strain,
    idx_t             ipoint ) const

{
  using jem::newInstance;

  using jem::io::PrintWriter;
  using jem::io::endl;


  Ref<PrintWriter> out = newInstance<PrintWriter>
    ( &jem::System::debug("j2mat") );

  out->nformat.setFractionDigits ( 10 );

  if ( ex.what().equals ( "No convergence" ) )
  {
    *out << "No convergence in return mapping algorithm " << endl;
  }
  else if ( ex.what().equals ( "nan" ) )
  {
    *out << "NaN detected in return mapping algorithm " << endl;
  }
  else if ( ex.what().equals ( "Negative dgam" ) )
  {
    *out << "Negative increment found" << endl;
  }
  else
  {
    *out << "Caught unkown exception: " << ex.what() << endl;
  }
}


//-----------------------------------------------------------------------
//   comp2DStiffMat_ 
//-----------------------------------------------------------------------


void J2Material::comp2DStiffMat_

  ()

{
  const idx_t  n  = FEMUtils::STRAIN_COUNTS[j2Rank_];

  const double E  = young   ();
  const double nu = poisson ();

  stiff2D_.resize ( n, n );
  stiff2D_ = 0.;

  if      ( j2Rank_ == 2_idx && state_ == ProblemType::PlaneStrain )
  {
    const double a = E / ((1. + nu) * (1. - 2. * nu));
    const double b = .5 * (1. - 2. * nu);
    const double c = 1. - nu;

    stiff2D_ (0,0) = a * c;
    stiff2D_ (0,1) = stiff2D_ (1,0) = a * nu;
    stiff2D_ (1,1) = a * c;
    stiff2D_ (2,2) = a * b;
  }
  else if ( j2Rank_ == 2_idx && state_ == ProblemType::PlaneStress )
  {
    const double a = E / (1. - nu * nu);

    stiff2D_ (0,0) = a;
    stiff2D_ (0,1) = stiff2D_ (1,0) = a * nu;
    stiff2D_ (1,1) = a;
    stiff2D_ (2,2) = a * .5 * (1. - nu);
  }
  else
  {
    throw jem::IllegalInputException ( 
      CLASS_NAME, "unexpected 2D material for local rank!" 
    );
  }
}


//-----------------------------------------------------------------------
//   reduce3DVector_
//-----------------------------------------------------------------------

 
void J2Material::reduce3DVector_
    
  ( const Vector&  v,
    const Vec6&    t ) const

{
  // Reduce a full 3D tuple to a 2D or 3D vector (works the same for both 
  // stress and strain)

  const idx_t n = v.size ();

  if       ( n == 3_idx )
  {
    v[0] = t[0];
    v[1] = t[1];
    v[2] = t[3];
  }
  else if ( n == 4_idx )
  {
    Tuple<idx_t,4> aximap ( 0, 1, 3, 2 );

    for ( idx_t i = 0; i < 4; ++i )
    {
      v[i] = t[aximap[i]];
    }
  }
  else
  {
    v[0] = t[0];
    v[1] = t[1];
    v[2] = t[2];
    v[3] = t[3];
    v[4] = t[4];
    v[5] = t[5];
  }
}


