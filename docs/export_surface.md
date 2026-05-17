# Export Surface Audit

PyNNS follows installed R NNS 12.0 behavior for the Python API it exposes. This
audit records R public exports that are implemented, intentionally renamed, or
explicitly guarded while dependencies remain unresolved.

## Implemented Or Renamed

| R export | PyNNS API | Notes |
|---|---|---|
| `Co.LPM`, `Co.UPM`, `D.LPM`, `D.UPM` | `co_lpm`, `co_upm`, `d_lpm`, `d_upm` | Pairwise partial-moment wrappers |
| `Co.LPM_nD`, `Co.UPM_nD`, `DPM_nD` | `co_lpm_nd`, `co_upm_nd`, `dpm_nd` | n-dimensional partial-moment wrappers |
| `LPM`, `UPM`, `LPM.ratio`, `UPM.ratio` | `lpm`, `upm`, `lpm_ratio`, `upm_ratio` | Core partial moments |
| `LPM.VaR`, `UPM.VaR` | `lpm_var`, `upm_var` | Deterministic VaR helpers |
| `NNS.ANOVA` | `nns_anova` | Binary and multi-group paths |
| `NNS.ARMA` | `nns_arma` | Forecast path |
| `NNS.ARMA.optim` | `nns_arma_optim` | Seasonal-factor optimizer |
| `NNS.boost` | `nns_boost` | Deterministic small-feature path |
| `NNS.caus` | `nns_causation` | Public causation wrapper |
| `NNS.CDF` | `nns_cdf` | Partial-moment CDF wrapper |
| `NNS.copula` | `nns_copula` | Copula dependence |
| `NNS.dep` | `nns_dep`, `nns_cor` | Dependence and correlation convenience |
| `NNS.diff` | `nns_diff` | Scalar callable differentiation |
| `NNS.distance`, `NNS.distance.bulk` | `nns_distance`, `nns_distance_bulk` | Distance helpers |
| `NNS.FSD`, `NNS.SSD`, `NNS.TSD` | `fsd`, `ssd`, `tsd` | Stochastic dominance |
| `NNS.FSD.uni`, `NNS.SSD.uni`, `NNS.TSD.uni` | `fsd_uni`, `ssd_uni`, `tsd_uni` | Unidirectional dominance helpers |
| `NNS.gravity`, `NNS.mode`, `NNS.rescale` | `nns_gravity`, `nns_mode`, `nns_rescale` | Central-tendency helpers |
| `NNS.MC`, `NNS.meboot` | `nns_mc`, `nns_meboot` | Stochastic paths use NumPy RNG |
| `NNS.M.reg` | `nns_m_reg` | Numeric internal path; raw factor expansion guarded |
| `NNS.moments`, `PM.matrix` | `nns_moments`, `pm_matrix` | Classical and PM matrix helpers |
| `NNS.norm` | `nns_norm` | Normalization wrapper |
| `NNS.part` | `nns_part` | Partition map |
| `NNS.reg` | `nns_reg` | Main regression wrapper |
| `NNS.SD.cluster`, `NNS.SD.efficient.set` | `nns_sd_cluster`, `sd_efficient_set` | SD clustering and efficient set |
| `NNS.seas` | `nns_seas` | Seasonality test |
| `NNS.SS` | `nns_ss` | Stochastic superiority |
| `NNS.stack` | `nns_stack` | Stack wrapper |
| `NNS.VAR` | `nns_var` | Numeric VAR wrapper with `dim_red_method` values `"cor"`, `"NNS.dep"`, `"NNS.caus"`, and `"all"` implemented |
| `NNS.nowcast` core | `nns_nowcast_panel`, `nns_nowcast(fetch=True, provider_backend=...)`, `pynns.providers.CsvNowcastProvider` | Deterministic user-supplied monthly panel and explicit provider path, including local CSV panels; no default live provider |
| `dy.dx` | `dy_dx` | `eval_point="overall"` and numeric `eval_point` vectors are implemented; multivariate form (`dy.d_`) |
| `dy.d_` | `dy_d` | Scalar `wrt` is implemented; vectorized `wrt` for `eval_points="mean"` and `mixed=False` is implemented; other vectorized modes remain deferred |
| `factor_2_dummy`, `factor_2_dummy_FR` | `factor_2_dummy`, `factor_2_dummy_fr` | Categorical expansion helpers |
| `ecdf.pm`, `mean.pm`, `var.pm`, `skew.pm`, `kurt.pm` | `ecdf_pm`, `mean_pm`, `var_pm`, `skew_pm`, `kurt_pm` | Partial-moment classical helpers |

## Guarded Public Exports And Methods

These names or method combinations are exported so callers get an explicit,
documented `NotImplementedError` instead of an accidental missing attribute.

| R export | PyNNS API | Blocker |
|---|---|---|
| `NNS.nowcast` default live fetching | `nns_nowcast` | No implicit FRED/Yahoo provider; call with `fetch=True` and an explicit provider backend |

## Internal Or Out Of Scope

`NNS.ANOVA.bin`, `NNS.M.reg`, `Uni.caus`, `NNS.caus.matrix`,
`NNS.dep.matrix`, compiled `*_cpp` shims, sampling helpers, and generated-vector
helpers are R internal surfaces. PyNNS exposes the public wrappers that depend
on them instead of mirroring every internal helper as a top-level API.
