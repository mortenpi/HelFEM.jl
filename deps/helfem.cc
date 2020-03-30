#include <armadillo>
#include "jlcxx/jlcxx.hpp"
#include "general/utils.h"
#include "general/scf_helpers.h"
#include "general/polynomial_basis.h"
#include "general/model_potential.h"
#include "atomic/basis.h"
#include <string>

std::string helfem_version() {
    return "vX.Y.Z";
}

void helfem_verbose(bool verbose) {
    if(verbose) {
        std::cout << "Setting HelFEM library to verbose mode." << std::endl;
    }
    helfem::verbose = verbose;
}

auto helfem_basis(int nnodes, int nelem, int primbas, double rmax, int igrid, double zexp, int nquad) {
    helfem::polynomial_basis::PolynomialBasis * poly = helfem::polynomial_basis::get_basis(primbas, nnodes);
    if(nquad <= 0) nquad = 5 * poly->get_nbf();
    return helfem::atomic::basis::RadialBasis(poly, nquad, nelem, rmax, igrid, zexp);
}

namespace jlcxx
{
  template<> struct SuperType<helfem::modelpotential::PointNucleus> { typedef helfem::modelpotential::ModelPotential type; };
  template<> struct SuperType<helfem::modelpotential::GaussianNucleus> { typedef helfem::modelpotential::ModelPotential type; };
  template<> struct SuperType<helfem::modelpotential::SphericalNucleus> { typedef helfem::modelpotential::ModelPotential type; };
  template<> struct SuperType<helfem::modelpotential::HollowNucleus> { typedef helfem::modelpotential::ModelPotential type; };
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    mod.method("version", &helfem_version);
    mod.method("verbose", &helfem_verbose);

    mod.add_type<arma::vec>("ArmaVector")
        .method("at", [] (const arma::vec& m, arma::uword i) { return m(i); })
        .method("nrows", [] (const arma::vec& m) { return m.n_rows; });

    mod.add_type<arma::mat>("ArmaMatrix")
        .method("at", [] (const arma::mat& m, arma::uword i, arma::uword j) { return m(i, j); })
        .method("nrows", [] (const arma::mat& m) { return m.n_rows; })
        .method("ncols", [] (const arma::mat& m) { return m.n_cols; });

    mod.method("form_Sinvh", &helfem::scf::form_Sinvh);

    mod.add_type<helfem::modelpotential::ModelPotential>("ModelPotential")
        .method("V", static_cast<double (helfem::modelpotential::ModelPotential::*)(double) const>(&helfem::modelpotential::ModelPotential::V));
    mod.add_type<helfem::modelpotential::PointNucleus>("PointNucleus", jlcxx::julia_base_type<helfem::modelpotential::ModelPotential>())
        .constructor<int>();
    mod.add_type<helfem::modelpotential::GaussianNucleus>("GaussianNucleus", jlcxx::julia_base_type<helfem::modelpotential::ModelPotential>())
        .constructor<int, double>();
    mod.add_type<helfem::modelpotential::SphericalNucleus>("SphericalNucleus", jlcxx::julia_base_type<helfem::modelpotential::ModelPotential>())
        .constructor<int, double>();
    mod.add_type<helfem::modelpotential::HollowNucleus>("HollowNucleus", jlcxx::julia_base_type<helfem::modelpotential::ModelPotential>())
        .constructor<int, double>();

    // mod.add_type<helfem::polynomial_basis::PolynomialBasis>("PolynomialBasis")
    //      .method("get_nbf", &helfem::polynomial_basis::PolynomialBasis::get_nbf)
    //      .method("get_noverlap", &helfem::polynomial_basis::PolynomialBasis::get_noverlap);
    mod.add_type<helfem::atomic::basis::RadialBasis>("RadialBasis")
        .method("nquad", &helfem::atomic::basis::RadialBasis::get_nquad)
        .method("nbf", &helfem::atomic::basis::RadialBasis::Nbf)
        .method("nel", &helfem::atomic::basis::RadialBasis::Nel)
        .method("radial_integral", static_cast<arma::mat (helfem::atomic::basis::RadialBasis::*)(const helfem::atomic::basis::RadialBasis &, int, bool, bool) const>(&helfem::atomic::basis::RadialBasis::radial_integral))
        .method("model_potential", static_cast<arma::mat (helfem::atomic::basis::RadialBasis::*)(const helfem::atomic::basis::RadialBasis &, const helfem::modelpotential::ModelPotential *, bool, bool) const>(&helfem::atomic::basis::RadialBasis::model_potential))
        .method("overlap", static_cast<arma::mat (helfem::atomic::basis::RadialBasis::*)(const helfem::atomic::basis::RadialBasis &) const>(&helfem::atomic::basis::RadialBasis::overlap));

    mod.method("basis", &helfem_basis);
}
