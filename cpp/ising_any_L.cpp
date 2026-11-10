// =============================================================================
//  2D Ising model, Metropolis Monte Carlo, simulated annealing
//  Parallelised over independent random seeds with OpenMP.
//
//  What the program does
//  ---------------------
//  For every seed k it
//    1. fills an L x L lattice with random +1/-1 spins (random "hot" start),
//    2. visits the temperatures in T_SCHEDULE from high to low (annealing).
//       At each temperature the lattice from the previous temperature is the
//       starting point, it is equilibrated, and then the magnetisation per
//       spin is recorded at regular intervals,
//    3. computes <m>, <|m|>, susceptibility and the Binder cumulant U4 from
//       those recorded values,
//    4. writes the results to CSV files (see "Output files").
//  The temperature loop of one seed is serial (each temperature starts from
//  the previous one), but different seeds are independent, so the seeds are
//  distributed over the CPU cores with OpenMP.
//
//  "MC step" in this file means ONE single-spin flip attempt, not a sweep.
//  One sweep = N = L*L steps.  All step counts below are in single-spin
//  attempts.
//
//  Optimisations (the results do not change, only the speed)
//  ---------------------------------------------------------
//   1. mt19937 random number generator.
//   2. Boltzmann factors exp(-4/T) and exp(-8/T) are computed once per
//      temperature, so there is no exp() call in the inner loop.
//   3. One random number per step gives the site (x, y).
//   4. Neighbour index tables remove the modulo operation from the inner loop.
//
//  This is the version for ANY lattice size (for example L = 40 or 80). For L a
//  power of 2 use ising.cpp instead: it is the same simulation and gives the same
//  results, but picks the random site with a bit shift and runs faster.
//
//  Parameters you may want to change
//  ---------------------------------
//  Set these when compiling (the Makefile passes them on, see README):
//     L_SIZE      lattice size L (any L >= 2, e.g. 16, 32, 40, 64, 80, 128)
//     SEED_START  first seed number
//     N_SEEDS     number of seeds in this run (seeds SEED_START ... SEED_START+N_SEEDS-1)
//     QUICK       define it for a very short schedule, used by the tests and
//                 for a quick check that the program works
//  The temperatures and the number of steps per temperature are set further
//  down in build_temp_array() and get_steps().
//
//  Compile (from the repository root, or just use `make`):
//     g++ -O3 -march=native -ffast-math -fopenmp -std=c++17 \
//         -DL_SIZE=64 -DSEED_START=1 -DN_SEEDS=8 -o build/ising_L64 cpp/ising_any_L.cpp
//  Run:
//     ./build/ising_L64 [output_directory]
//  Number of threads (use at most the number of PHYSICAL cores):
//     OMP_NUM_THREADS=4 ./build/ising_L64 runs/L64
//
//  Output files (k = seed number), written to the output directory
//  ----------------------------------------------------------------
//     Run_no_<L>_<k>_Dist_Curve.csv  every recorded m, one column per temperature
//                                    (needed for the magnetisation distribution curves)
//     m_<L>_<k>.csv                  mean magnetisation <m>      per temperature
//     Abs_m_<L>_<k>.csv              mean |m|                    per temperature
//     Sus_<L>_<k>.csv                susceptibility              per temperature
//     U_<L>_<k>.csv                  Binder cumulant U4          per temperature
//  The rows of the per-temperature files follow the order of T_SCHEDULE
//  (highest temperature first).  Use scripts/average_seeds.py to average the
//  seeds.
// =============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <omp.h>

// --- Tunable constants --------------------------------------------------------
#ifndef L_SIZE
#define L_SIZE 64
#endif
#ifndef SEED_START
#define SEED_START 1
#endif
#ifndef N_SEEDS
#define N_SEEDS 8
#endif

static constexpr int L          = L_SIZE;
static constexpr int N          = L * L;
static constexpr int SEED_FIRST = SEED_START;
static constexpr int SEED_COUNT = N_SEEDS;

// --- Choosing a random site ---------------------------------------------------
//  One 32-bit random number r gives the site index idx = floor(r * N / 2^32),
//  a multiplication and a shift instead of a modulo. For L a power of 2 this is
//  exactly the top log2(N) bits of r, so every site has exactly the same
//  probability. For other L (for example 40 or 80) the probabilities differ
//  from 1/N by less than N / 2^32 (about 1e-6 for L = 80), which is negligible.
//  Row and column follow from idx / L and idx % L; for a power of 2 the compiler
//  turns these into a shift and a mask.
static_assert(L >= 2 && static_cast<long long>(L) * L <= (1LL << 31), "L out of range (need 2 <= L <= 46340)");

// =============================================================================
//  Random numbers (Mersenne Twister, one generator per seed)
// =============================================================================
struct RNG {
    std::mt19937 gen;
    explicit RNG(uint32_t seed) : gen(seed) {}

    inline uint32_t next32() { return gen(); }                  // raw 32-bit word

    // Uniform site index in [0, N)
    inline int next_site() { return static_cast<int>((static_cast<uint64_t>(next32()) * N) >> 32); }

    // Uniform double in [0, 1)  (27 random bits)
    inline double next_double() { return (next32() >> 5) * (1.0 / (1u << 27)); }
};

// =============================================================================
//  Temperature schedule (T_SCHEDULE): from high T to low T, 131 points.
//    4.5  ... 2.6   in steps of 0.10   (20 points, far above Tc)
//    2.5  ... 1.46  in steps of 0.01   (105 points, dense around Tc = 2.269)
//    1.25 ... 0.5   in steps of 0.15   (6 points, far below Tc)
//  To change the temperatures edit the three loops below.
// =============================================================================
static inline double r2(double x) { return std::round(x * 100.0) / 100.0; }

std::vector<double> build_temp_array() {
    std::vector<double> T;
    for (double t = 4.5;  ; t -= 0.10) { double v = r2(t); if (v <= 2.5)  break; T.push_back(v); }
    for (double t = 2.5;  ; t -= 0.01) { double v = r2(t); if (v <= 1.45) break; T.push_back(v); }
    for (double t = 1.25; ; t -= 0.15) { double v = r2(t); if (v <= 0.45) break; T.push_back(v); }
    return T;
}

// =============================================================================
//  Number of steps per temperature
//    eq   steps discarded for equilibration
//    mc   total number of steps at this temperature
//    mea  a measurement is taken every `mea` steps after the first `eq`
//  Number of measurements per temperature = (mc - eq) / mea = N_MEAS (5000).
//  All numbers are single-spin flip attempts; one sweep (MC sweep) = L*L attempts.
//
//  Close to Tc (2.0 <= T <= 2.5) the system relaxes very slowly, so it gets by
//  far the longest equilibration and the longest gap between measurements.
//
//  Presets (the values used for the dissertation data) exist for L = 16, 32, 40,
//  64, 80 and 128. Each preset is given as (equilibration, measurement interval)
//  for the three temperature ranges; mc = eq + N_MEAS * mea.
//
//  Any other L: the L = 64 values are scaled up or down with
//        value(L) = value(64) * (L / 64)^(2 + z)
//  where z is the dynamic exponent of the Metropolis algorithm: z = 2.2 close to
//  Tc (2.0 <= T <= 2.5) and z = 2 elsewhere. The 2 comes from counting steps
//  as single flip attempts (L*L of them make one sweep); L^z is the growth of
//  the relaxation time itself. This is only a starting point: check the
//  equilibration for your L (for example by plotting the energy against the
//  step number) and edit the numbers if needed.
//  To change any of the settings edit the numbers below.
// =============================================================================
struct TSteps { long long eq, mc, mea; };

constexpr long long N_MEAS = 5000;       // measurements per temperature

// eq and mea given, total number of steps follows from the number of measurements
constexpr TSteps with_meas(long long eq, long long mea) { return {eq, eq + N_MEAS * mea, mea}; }

TSteps get_steps(double T) {
#ifdef QUICK
    // Very short schedule: 2000 measurements, 100 attempts apart. For tests only.
    (void)T;
    return {20000LL, 20000LL + 2000LL * 100LL, 100LL};
#else
    const int range = (T > 2.5) ? 0 : (T >= 2.0 ? 1 : 2);   // 0: above, 1: near Tc, 2: below
    //                                 T > 2.5                    2.0 <= T <= 2.5               T < 2.0
    //                                 eq,         mea           eq,           mea            eq,        mea
    if constexpr (L == 16) {
        constexpr TSteps p[3] = { with_meas(500000,   500), with_meas(7000000,    25000),  with_meas(300000,  1000) };
        return p[range];
    } else if constexpr (L == 32) {
        constexpr TSteps p[3] = { with_meas(500000,   500), with_meas(30000000,   25000),  with_meas(300000,  1000) };
        return p[range];
    } else if constexpr (L == 40) {
        constexpr TSteps p[3] = { with_meas(500000,  1000), with_meas(50000000,   50000),  with_meas(300000,  1000) };
        return p[range];
    } else if constexpr (L == 64) {
        constexpr TSteps p[3] = { with_meas(500000,   500), with_meas(50000000,  100000),  with_meas(300000,  1000) };
        return p[range];
    } else if constexpr (L == 80) {
        constexpr TSteps p[3] = { with_meas(3000000, 31000), with_meas(200000000, 250000),  with_meas(3000000, 15000) };
        return p[range];
    } else if constexpr (L == 128) {
        constexpr TSteps p[3] = { with_meas(3000000, 100000), with_meas(300000000, 2000000), with_meas(3000000, 50000) };
        return p[range];
    } else {
        // Any other L: scale the L = 64 values (see the comment above).
        const long long eq64[3]  = {500000, 50000000, 300000};
        const long long mea64[3] = {500,    100000,   1000};
        const double z      = (range == 1) ? 2.2 : 2.0;
        const double factor = std::pow(L / 64.0, 2.0 + z);
        const long long mea = std::max(1LL, std::llround(mea64[range] * factor));
        return with_meas(std::llround(eq64[range] * factor), mea);
    }
#endif
}

// =============================================================================
//  CSV writers
// =============================================================================
void write_csv_column(const std::string &filename, const std::string &col_name,
                      const std::vector<double> &data) {
    std::ofstream f(filename);
    f << col_name << "\n" << std::setprecision(15);
    for (double v : data) f << v << "\n";
}

// One column per temperature (header = the temperature), one row per measurement
void write_dist_csv(const std::string &filename, const std::vector<double> &Temp,
                    const std::vector<std::vector<double>> &dist) {
    std::ofstream f(filename);
    f << std::setprecision(15);
    for (size_t t = 0; t < Temp.size(); ++t) {
        if (t) f << ",";
        std::ostringstream oss; oss << std::setprecision(10) << Temp[t];
        f << oss.str();
    }
    f << "\n";
    size_t n_rows = dist.empty() ? 0 : dist[0].size();
    for (size_t r = 0; r < n_rows; ++r) {
        for (size_t t = 0; t < dist.size(); ++t) {
            if (t) f << ",";
            f << dist[t][r];
        }
        f << "\n";
    }
}

// =============================================================================
//  One seed: anneal through all temperatures and write the output files
// =============================================================================
void run_seed(int k, const std::vector<double> &Temp, const std::vector<TSteps> &steps,
              const std::string &outdir) {

    const int n_T = static_cast<int>(Temp.size());

    // The seed number is the RNG seed.  The first N random bits give the initial spins.
    RNG rng(static_cast<uint32_t>(k));
    std::vector<int> state(N);
    for (int i = 0; i < N; ++i) state[i] = (rng.next32() & 1) ? 1 : -1;

    // Periodic neighbours: right/left of index i
    int nb_right[L], nb_left[L];
    for (int i = 0; i < L; ++i) {
        nb_right[i] = (i + 1) % L;
        nb_left[i]  = (i - 1 + L) % L;
    }

    std::vector<std::vector<double>> Total_Mag_per_Temp(n_T);

    // --- Temperature loop (serial: each temperature starts from the previous state) ---
    for (int t = 0; t < n_T; ++t) {

        const TSteps &st  = steps[t];
        const double beta = 1.0 / Temp[t];      // k_B = J = 1

        // In the 2D Ising model deltaE can only be -8, -4, 0, +4, +8.
        // exp() is only needed for deltaE = 4 and 8, so precompute both.
        const double exp_neg4 = std::exp(-4.0 * beta);
        const double exp_neg8 = std::exp(-8.0 * beta);

        // Number of multiples of `mea` in [eq, mc). This equals (mc - eq) / mea
        // when eq is a multiple of mea, and stays correct when it is not.
        const long long n_meas = (st.mc - 1) / st.mea - (st.eq + st.mea - 1) / st.mea + 1;
        std::vector<double> Total_Mag(n_meas, 0.0);

        // Total magnetisation M, updated incrementally instead of re-summed.
        long long Mag = 0;
        for (int s : state) Mag += s;

        long long meas = 0;

        // --- Metropolis loop ---------------------------------------------------
        for (long long i = 0; i < st.mc; ++i) {

            const int idx = rng.next_site();     // random site
            const int x   = idx / L;             // row
            const int y   = idx % L;             // column

            // Sum of the four nearest neighbours
            const int sigma_j = state[x           * L + nb_right[y]]
                              + state[x           * L + nb_left[y]]
                              + state[nb_left[x]  * L + y]
                              + state[nb_right[x] * L + y];

            const int s_i    = state[idx];
            const int deltaE = 2 * sigma_j * s_i;   // energy change if the spin flips (J = 1)

            // Metropolis rule
            bool accept;
            if      (deltaE <= 0)  accept = true;
            else if (deltaE == 4)  accept = rng.next_double() < exp_neg4;
            else                   accept = rng.next_double() < exp_neg8;

            if (accept) {
                state[idx] = -s_i;
                Mag += 2 * state[idx];   // new spin is -s_i, so M changes by -2*s_i = +2*(new spin)
            }

            // Record the magnetisation per spin every `mea` steps after equilibration
            if (i >= st.eq && (i % st.mea) == 0)
                Total_Mag[meas++] = static_cast<double>(Mag) / N;
        }

        Total_Mag_per_Temp[t] = std::move(Total_Mag);
    }

    // --- Observables from the recorded magnetisations ---------------------------
    std::vector<double> m_out(n_T), sus_out(n_T), U_out(n_T), Abs_m_out(n_T);

    for (int t = 0; t < n_T; ++t) {
        const auto &mag = Total_Mag_per_Temp[t];
        const double cnt = static_cast<double>(mag.size());

        double mean_m = 0, mean_abs = 0, mean_m2 = 0, mean_m4 = 0;
        for (double v : mag) {
            mean_m   += v;
            mean_abs += std::abs(v);
            mean_m2  += v * v;
            mean_m4  += v * v * v * v;
        }
        mean_m /= cnt; mean_abs /= cnt; mean_m2 /= cnt; mean_m4 /= cnt;

        // chi = N (<m^2> - <|m|>^2) / T      (k_B = 1)
        sus_out[t]   = (1.0 / Temp[t]) * N * (mean_m2 - mean_abs * mean_abs);
        Abs_m_out[t] = mean_abs;
        // Binder cumulant U4 = 1 - <m^4> / (3 <m^2>^2)
        U_out[t]     = 1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2);
        m_out[t]     = mean_m;
    }

    // --- Write output ---------------------------------------------------------------
    const std::string Ls = std::to_string(L);
    const std::string ks = std::to_string(k);
    const std::string p  = outdir.empty() ? "" : outdir + "/";

    write_dist_csv(  p + "Run_no_" + Ls + "_" + ks + "_Dist_Curve.csv", Temp, Total_Mag_per_Temp);
    write_csv_column(p + "m_"      + Ls + "_" + ks + ".csv", "m_"      + Ls + "_" + ks, m_out);
    write_csv_column(p + "Sus_"    + Ls + "_" + ks + ".csv", "Sus_"    + Ls + "_" + ks, sus_out);
    write_csv_column(p + "U_"      + Ls + "_" + ks + ".csv", "U_"      + Ls + "_" + ks, U_out);
    write_csv_column(p + "Abs_m_"  + Ls + "_" + ks + ".csv", "Abs_m_"  + Ls + "_" + ks, Abs_m_out);
}

// =============================================================================
//  Main
// =============================================================================
int main(int argc, char **argv) {
    const std::string outdir = (argc > 1) ? argv[1] : ".";
    std::filesystem::create_directories(outdir);

    std::vector<double> Temp  = build_temp_array();
    const int           n_T   = static_cast<int>(Temp.size());
    std::vector<TSteps> steps(n_T);
    for (int t = 0; t < n_T; ++t) steps[t] = get_steps(Temp[t]);

    long long attempts = 0;
    for (const TSteps &st : steps) attempts += st.mc;

    std::cout << "Lattice            : " << L << " x " << L << "\n";
    std::cout << "Temperature points : " << n_T << "\n";
    std::cout << "Attempts per seed  : " << attempts << " single-spin flip attempts (use this to estimate the run time)\n";
    std::cout << "Seeds to run       : " << SEED_FIRST << " to " << (SEED_FIRST + SEED_COUNT - 1) << "\n";
    std::cout << "OpenMP threads     : " << omp_get_max_threads() << "\n";
    std::cout << "Output directory   : " << outdir << "\n";
    std::cout << "Starting...\n" << std::flush;

    // Each seed is independent, so the seeds are shared between the threads.
    // schedule(dynamic, 1): a thread that finishes a seed takes the next free one.
    #pragma omp parallel for schedule(dynamic, 1) default(none) shared(Temp, steps, outdir, std::cout)
    for (int k = SEED_FIRST; k < SEED_FIRST + SEED_COUNT; ++k) {
        run_seed(k, Temp, steps, outdir);
        #pragma omp critical
        std::cout << "  Seed " << k << " done (thread " << omp_get_thread_num() << ")\n" << std::flush;
    }

    std::cout << "All seeds complete.\n";
    return 0;
}
