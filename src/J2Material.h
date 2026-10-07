
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


#ifndef J2_MATERIAL_H
#define J2_MATERIAL_H


#include <jem/base/Exception.h>
#include <jem/numeric/func/Function.h>
#include <jem/util/Flex.h>


#include "IsotropicMaterial.h"
#include "Invariants.h"


using jem::Exception;

using jem::util::Flex;
using jem::numeric::Function;


//=======================================================================
//   foward-declaration
//=======================================================================
//
//   This class is defined to handle Plane Stress, Plane
//   Strain and 3D cases.


class YieldFunc_;


//=======================================================================
//   class J2Material
//=======================================================================


class J2Material : public IsotropicMaterial 
{
 public:

  JEM_DECLARE_CLASS       ( J2Material, IsotropicMaterial );

  static const char*        TYPE_NAME;

  static const char*        RM_TOLERANCE;
  static const char*        RM_MAX_ITER;
  static const char*        YIELD_PROP;


  explicit                  J2Material

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat )                    override;

  virtual void              getConfig 

    ( const Properties&       conf,
      const Properties&       globdat )              const override; 

  virtual void              commit                ()       override;

  virtual void              cancel                ()       override;

  virtual void              allocPoints

    ( idx_t                   count )                      override;

  virtual idx_t             rank                  () const noexcept
                                                           override;

  virtual ProblemType       getProblemType        () const noexcept 
                                                           override;

  virtual void              elasticUpdate

    ( const Vector&           stress,
      const Matrix&           stiff,
      const Vector&           strain )               const override;

  virtual Vector            elasticUpdate

    ( const Vector&           strain )               const override;

  virtual void              update 

    ( const Vector&           stress,
      const Matrix&           stiff,
      const Vector&           strain,
      idx_t                   ip )                   const override;

  virtual void              update 

    ( const Vector&           stress,
      const Vector&           strain,
      idx_t                   ip )                   const override;

  virtual Matrix            getStiffMat           () const override;

  static Ref<Material>      makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare               ();


 protected:

  virtual                  ~J2Material            ();


 private:

  class                     Hist_
  {
   public:
                            Hist_                 (); 

    Vec6                    epsp;
    double                  epspeq;
    double                  dissipation;
    bool                    loading;
  };

  Vec6                      tmatmul_

    ( const Matrix&           mat,
      const Vec6&             tuple )                const;

  inline void               t6_to_v6_

    ( const Vector&           t6,
      const Vec6&             v6 )                   const;

  inline void               v6_to_t6_

    (       Vec6&             t6,
      const Vector&           v6 )                   const;

  inline Vec6               v6_to_t6_

    ( const Vector&           v6 )                   const;

  Ref<Function>             makeFunc_

    ( const String&           name, 
      const Properties&       props,
      const Properties&       globdat )              const;

  void                      handleException_ 

    ( const Exception&        ex,
      const Vector&           strain,
      idx_t                   ipoint )               const;

  void                      comp2DStiffMat_       ();

  void                      reduce3DVector_
    
    ( const Vector&           v,
      const Vec6&             t )                    const; 


 private:

  Matrix                    stiff2D_;

  Flex<Hist_>               preHist_;
  Flex<Hist_>               newHist_;
  Flex<Hist_>*              latestHist_;

  Vector                    v61_;
  Vector                    v62_;

  String                    stateString_;

  // Hardening functions.

  Ref<YieldFunc_>           y_;

  double                    rmTolerance_;

  idx_t                     rmMaxIter_;

  // Need own rank, because base rank is always 3, for 3D stiff.

  idx_t                     j2Rank_;

  ProblemType               state_;

};


//=======================================================================
//   class YieldFunc_ (Helper class)
//=======================================================================
//
//   General class to handle yield functions for all 
//   assumptions: Plane Stress, Plane Strain, 3D.


class YieldFunc_ : public jem::Object
{
 public:

                            YieldFunc_  
      
    ( double                  young,
      double                  poisson,
      const Ref<Function>     sigY );

  void                      setRmSettings 

    ( double                  rmTolerance, 
      idx_t                   rmMaxIter );

  void                      setElasticStiff

    ( const Matrix&           mat );

  double                    findRoot 
      
    ( double                  dgam0 );

  void                      findBounds

    ( double&                 gmin,
      double&                 gmax,
      double&                 fmin,
      double&                 fmax );

  double                    findRoot

   ( double                   gmin,
     double                   gmax,
     double                   fmin,
     double                   fmax );

  virtual bool              isLinear

    ( const Vec6&             sigtr,
      double                  epspeq0 )                    = 0;

  virtual void              compSigUpdated

    ( Vec6&                   sig, 
      double                  dgam, 
      const Vec6&             sigtr )                      = 0;

  virtual void              compPlastStrInc

    ( Vec6&                   depsp, 
      double                  dgam )                       = 0;

  virtual double            compEquPlastInc

    ( double                  dgam )                       = 0;

  virtual Matrix            compTangStiff

    ( const double            dgam )                       = 0;

  inline Ref<Function>      getSigmaYFunc         () const;
  inline double             getEpspeq             () const;
  inline Matrix             getElastStiff         () const;

 protected:

  virtual                  ~YieldFunc_            ();


 protected:

  virtual double            evalTrial_            () const = 0;

  virtual double            eval_

    ( double                  dgam )                       = 0;

  virtual double            evalDer_

    ( double                  dgam )                       = 0;

  virtual double            evalKsi_

    ( double                  dgam )                       = 0; 

  virtual double            evalDerKsi_

    ( double                  dgam )                       = 0; 

  Vec6                      deviatoric_

    ( const Vec6&             full,
      const double            p )                    const;


  Ref<Function>             sigmaY_;
  Properties                globdat_;

  Matrix                    De_;
  double                    rmTol_;
  double                    E_;
  double                    nu_;
  double                    G_;
  double                    K_;

  double                    epspeq0_;
  double                    epspeq_;
  double                    ksitr_;

  idx_t                     maxIter_;
  idx_t                     rankDep_;  // rank of elastoplastic matrix

 private:

  double                    estimateRoot_

    ( double                  gmin,
      double                  gmax,
      double                  fmin,
      double                  fmax )                 const;

  void                      improveBounds_

    ( double&                 gmin,
      double&                 gmax,
      double&                 fmin,
      double&                 fmax );

};


//=======================================================================
//   class YieldFunc3D_ (Helper class)
//=======================================================================
//
//   This class deals with 3D and Plane Strain problems.


class YieldFunc3D_ : public YieldFunc_
{
 public:

                            YieldFunc3D_  
      
    ( double                  young,
      double                  poisson,
      const Ref<Function>     yield    );

  bool                      isLinear

    ( const Vec6&             sigtr,
      double                  epspeq0 )                    override;

  void                      compSigUpdated

    ( Vec6&                   sig, 
      double                  dgam, 
      const Vec6&             sigtr )                      override;

  void                      compPlastStrInc

    ( Vec6&                   depsp, 
      double                  dgam )                       override;

  double                    compEquPlastInc

    ( double                  dgam )                       override;

  Matrix                    compTangStiff

    ( const double            dgam )                       override;

 protected:

  virtual                  ~YieldFunc3D_          ();

  double                    evalTrial_            () const override;

  double                    eval_

    ( double                  dgam )                       override;

  double                    evalDer_

    ( double                  dgam )                       override; 

  double                    evalKsi_

    ( double                  dgam )                       override; 

  double                    evalDerKsi_

    ( double                  dgam )                       override; 

 
  Vec6                      sigDtr_;                 
  Vec6                      sigD_;
  double                    ptrial_;

 private:

  Tuple<double,6,6>         getIdMatrix_          ();

  double                    normSig_

    ( const Vec6&             sig )                  const;

  void                      m66_to_m33_     
  
    ( const Matrix&           m33,
      const Matrix&           m66 )                  const;
};


//=======================================================================
//   class YieldFuncPS_ (Helper class)
//=======================================================================
//
//   This class deals with Plane Stress problems.


class YieldFuncPS_ : public YieldFunc_
{
 public:

                            YieldFuncPS_  
      
    ( double                  young,
      double                  poisson,
      const Ref<Function>     yield    );

  bool                      isLinear

    ( const Vec6&             sigtr,
      double                  epspeq0 )                    override;

  void                      compSigUpdated

    ( Vec6&                   sig, 
      double                  dgam, 
      const Vec6&             sigtr )                      override;

  void                      compPlastStrInc

    ( Vec6&                   depsp, 
      double                  dgam )                       override;

  double                    compEquPlastInc

    ( double                  dgam )                       override;

  Matrix                    compTangStiff

    ( double                  dgam )                       override;

 protected:

  virtual                  ~YieldFuncPS_          ();

  double                    evalTrial_            () const override;

  double                    eval_

    ( double                  dgam )                       override;

  double                    evalDer_

    ( double                  dgam )                       override; 

  double                    evalKsi_

    ( double                  dgam )                       override; 

  double                    evalDerKsi_

    ( double                  dgam )                       override; 


  Tuple<double,3,3>         P_;
  Vec3                      stress_;
  double                    a1tr_;
  double                    a2tr_;
  double                    a3tr_;

 private:

  Tuple<double,3,3>         getAMatrix_

    ( double                  dgam )                 const; 

  Tuple<double,3,3>         getEMatrix_

    ( double                  dgam )                 const; 

  inline void               t6_to_t3_

    (       Vec3&             t3,
      const Vec6&             t6 )                   const;

  inline void               t3_to_t6_

    (       Vec6&             t6,
      const Vec3&             t3 )                   const;

  inline void               ep3_to_ep6_

    (       Vec6&             t6,
      const Vec3&             t3 )                   const;

  void                      t33_to_m33_

    ( const Matrix&           m,
      const Tuple<double,3,3>& 
                              t )                    const; 
};


//#######################################################################
//   Implementation
//#######################################################################


//-----------------------------------------------------------------------
//   J2Material::v6_to_t6_
//-----------------------------------------------------------------------


void J2Material::v6_to_t6_

  (       Vec6&   t6,
    const Vector& v6 ) const

{
  t6[0] = v6[0]; t6[1] = v6[1]; t6[2] = v6[2]; 
  t6[3] = v6[3]; t6[4] = v6[4]; t6[5] = v6[5]; 
}


Vec6 J2Material::v6_to_t6_

  ( const Vector& v6 ) const

{
  Vec6 t6;
  v6_to_t6_ ( t6, v6 );
  return t6;
}


//-----------------------------------------------------------------------
//   J2Material::t6_to_v6_
//-----------------------------------------------------------------------


void J2Material::t6_to_v6_

  ( const Vector& v6,
    const Vec6&   t6 ) const

{
  v6[0] = t6[0]; v6[1] = t6[1]; v6[2] = t6[2]; 
  v6[3] = t6[3]; v6[4] = t6[4]; v6[5] = t6[5]; 
}


//-----------------------------------------------------------------------
//   YieldFunc_::getSigmaYFunc
//-----------------------------------------------------------------------


inline Ref<Function> YieldFunc_::getSigmaYFunc

  () const

{
  return sigmaY_;
}


//-----------------------------------------------------------------------
//   YieldFunc_::getEpspeq
//-----------------------------------------------------------------------


inline double YieldFunc_::getEpspeq

  () const

{
  return epspeq_;
}


//-----------------------------------------------------------------------
//   YieldFunc_::getElastStiff
//-----------------------------------------------------------------------


inline Matrix YieldFunc_::getElastStiff   

  () const

{
  return De_;
}


//-----------------------------------------------------------------------
//   YieldFuncPS_::t6_to_t3_ 
//-----------------------------------------------------------------------


void YieldFuncPS_::t6_to_t3_

  (       Vec3&  t3,
    const Vec6&  t6 ) const

{
  t3[0] = t6[0]; 
  t3[1] = t6[1]; 
  t3[2] = t6[3]; 
}


//-----------------------------------------------------------------------
//   YieldFuncPS_::t3_to_t6_ 
//-----------------------------------------------------------------------


void YieldFuncPS_::t3_to_t6_

  (       Vec6&  t6,
    const Vec3&  t3 ) const

{
  t6[0] = t3[0]; 
  t6[1] = t3[1]; 
  t6[2] = 0.; 
  t6[3] = t3[2]; 
  t6[4] = 0.; 
  t6[5] = 0.; 
}


//-----------------------------------------------------------------------
//   YieldFuncPS_::ep3_to_ep6_ 
//-----------------------------------------------------------------------
// This member takes into account the out-of-plane component of plastic
// strain.


void YieldFuncPS_::ep3_to_ep6_

  (       Vec6&  ep6,
    const Vec3&  ep3 ) const

{
  ep6[0] = ep3[0]; 
  ep6[1] = ep3[1]; 
  ep6[2] = -ep3[0]-ep3[1]; // out-of-plane component 
  ep6[3] = ep3[2]; 
  ep6[4] = 0.; 
  ep6[5] = 0.; 
}


#endif
