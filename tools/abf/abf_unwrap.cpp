// abf_unwrap.cpp
//
// Unwrap a mesh along pre-marked seam edges using geogram's ABF++
// (= MeshTailor paper ref [35], Sheffer/Levy 2005).
//
// Pipeline:
//   1. segment facets into charts by flood-fill that does not cross seam edges
//   2. for each chart, build a sub-mesh and flatten it with mesh_compute_ABF_plus_plus
//   3. grid-pack the charts into [0,1]^2
//   4. write uv.obj (v, vt, f v/vt) compatible with eval/uv_metrics.py
//
// Usage: abf_unwrap <mesh.obj> <seam.json> <uv.obj>

#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/attributes.h>
#include <geogram/basic/geometry.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/parameterization/mesh_ABF.h>
#include <geogram/parameterization/mesh_LSCM.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace GEO;

typedef std::pair<index_t, index_t> Edge;

static inline Edge mk_edge(index_t a, index_t b) {
    return Edge(std::min(a, b), std::max(a, b));
}

// Extract all non-negative integers from the JSON text and pair them up.
// Works for the flat "[[v0,v1],[v0,v1],...]" format used by the eval pipeline.
static std::set<Edge> read_seams(const std::string& path, bool& ok) {
    std::set<Edge> seams;
    ok = false;
    std::ifstream f(path);
    if (!f) return seams;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<long long> nums;
    size_t i = 0;
    while (i < s.size()) {
        char c = s[i];
        if (c == '-' || (c >= '0' && c <= '9')) {
            bool neg = (c == '-');
            size_t j = i + (neg ? 1 : 0);
            long long val = 0;
            bool any = false;
            while (j < s.size() && s[j] >= '0' && s[j] <= '9') {
                val = val * 10 + (s[j] - '0');
                j++; any = true;
            }
            if (any) { nums.push_back(neg ? -val : val); i = j; continue; }
        }
        i++;
    }
    for (size_t k = 0; k + 1 < nums.size(); k += 2) {
        if (nums[k] >= 0 && nums[k + 1] >= 0) {
            seams.insert(mk_edge((index_t)nums[k], (index_t)nums[k + 1]));
        }
    }
    ok = true;
    return seams;
}

// Total unsigned UV triangle area of a mesh (tex_coord is a VERTEX attribute).
static double mesh_uv_area(Mesh& M, Attribute<double>& uv) {
    if (!uv.is_bound() || uv.dimension() != 2) return 0.0;
    double s = 0.0;
    for (index_t f = 0; f < M.facets.nb(); f++) {
        index_t a = M.facets.vertex(f, 0);
        index_t b = M.facets.vertex(f, 1);
        index_t d = M.facets.vertex(f, 2);
        double ux0 = uv[2*a], uy0 = uv[2*a+1];
        double ux1 = uv[2*b], uy1 = uv[2*b+1];
        double ux2 = uv[2*d], uy2 = uv[2*d+1];
        s += 0.5 * std::abs((ux1-ux0)*(uy2-uy0) - (ux2-ux0)*(uy1-uy0));
    }
    return s;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <mesh.obj> <seam.json> <uv.obj>\n";
        return 1;
    }
    const char* mesh_path = argv[1];
    const char* seam_path = argv[2];
    const char* out_path = argv[3];

    GEO::initialize();
    GEO::Logger::instance()->set_quiet(true);

    Mesh mesh;
    if (!mesh_load(mesh_path, mesh)) {
        std::cerr << "Error: failed to load " << mesh_path << "\n";
        return 2;
    }

    bool seam_ok = false;
    std::set<Edge> seams = read_seams(seam_path, seam_ok);
    if (!seam_ok) std::cerr << "Warning: could not read seam file " << seam_path << "\n";

    // edge -> incident facets
    std::map<Edge, std::vector<index_t>> edge_facets;
    for (index_t f = 0; f < mesh.facets.nb(); f++) {
        index_t nbf = mesh.facets.nb_vertices(f);
        for (index_t lv = 0; lv < nbf; lv++) {
            index_t v1 = mesh.facets.vertex(f, lv);
            index_t v2 = mesh.facets.vertex(f, (lv + 1) % nbf);
            edge_facets[mk_edge(v1, v2)].push_back(f);
        }
    }

    // Flood-fill facets into charts, not crossing seam edges.
    std::vector<index_t> chart(mesh.facets.nb(), index_t(-1));
    index_t ncharts = 0;
    for (index_t start = 0; start < mesh.facets.nb(); start++) {
        if (chart[start] != index_t(-1)) continue;
        std::queue<index_t> q;
        q.push(start);
        chart[start] = ncharts;
        while (!q.empty()) {
            index_t f = q.front(); q.pop();
            index_t nbf = mesh.facets.nb_vertices(f);
            for (index_t lv = 0; lv < nbf; lv++) {
                index_t v1 = mesh.facets.vertex(f, lv);
                index_t v2 = mesh.facets.vertex(f, (lv + 1) % nbf);
                Edge e = mk_edge(v1, v2);
                if (seams.count(e)) continue;  // do not cross seam
                std::map<Edge, std::vector<index_t>>::iterator it = edge_facets.find(e);
                if (it == edge_facets.end()) continue;
                for (size_t k = 0; k < it->second.size(); k++) {
                    index_t nb = it->second[k];
                    if (nb != f && chart[nb] == index_t(-1)) {
                        chart[nb] = ncharts;
                        q.push(nb);
                    }
                }
            }
        }
        ncharts++;
    }

    // Per-chart flatten with ABF++ (fallback LSCM). chart_uv[c][orig_vertex]=(u,v)
    std::vector<std::map<index_t, std::pair<double, double>>> chart_uv(ncharts);
    index_t abf_ok = 0, abf_fail = 0;
    for (index_t c = 0; c < ncharts; c++) {
        std::vector<index_t> cf;
        for (index_t f = 0; f < mesh.facets.nb(); f++)
            if (chart[f] == c) cf.push_back(f);
        if (cf.empty()) continue;

        // build sub-mesh
        Mesh sub;
        std::map<index_t, index_t> vmap;
        std::vector<index_t> origv;
        for (size_t i = 0; i < cf.size(); i++) {
            index_t f = cf[i];
            for (index_t lv = 0; lv < mesh.facets.nb_vertices(f); lv++) {
                index_t ov = mesh.facets.vertex(f, lv);
                if (vmap.find(ov) == vmap.end()) {
                    index_t nv = sub.vertices.create_vertex();
                    auto p = mesh.vertices.point(ov);
                    sub.vertices.point(nv) = vec3(p[0], p[1], p[2]);
                    vmap[ov] = nv;
                    origv.push_back(ov);
                }
            }
        }
        for (size_t i = 0; i < cf.size(); i++) {
            index_t f = cf[i];
            index_t nbf = mesh.facets.nb_vertices(f);
            if (nbf != 3) { std::cerr << "Error: non-tri facet\n"; return 5; }
            sub.facets.create_triangle(
                vmap[mesh.facets.vertex(f, 0)],
                vmap[mesh.facets.vertex(f, 1)],
                vmap[mesh.facets.vertex(f, 2)]);
        }

        // Flatten: ABF++ (real [35]); fall back to LSCM if ABF++ throws or
        // collapses (degenerate, near-zero UV area — happens on some charts
        // with awkward topology from model-predicted seams).
        bool ok = false;
        try {
            mesh_compute_ABF_plus_plus(sub, "tex_coord", false);
            Attribute<double> uvt(sub.vertices.attributes(), "tex_coord");
            ok = (mesh_uv_area(sub, uvt) > 1e-12);
        } catch (...) { ok = false; }
        if (!ok) {
            try { mesh_compute_LSCM(sub, "tex_coord", false); } catch (...) {}
        }
        Attribute<double> uv(sub.vertices.attributes(), "tex_coord");
        if (mesh_uv_area(sub, uv) > 1e-12) {
            abf_ok++;
        } else {
            abf_fail++;
            continue;  // leave chart_uv[c] empty -> its triangles get vt 0,0
        }
        if (uv.is_bound() && uv.dimension() == 2) {
            // Per-chart area normalization: scale UVs so sum(UV area) == sum(3D area).
            // This makes the pooled std(log A_uv/A_3d) a pure distortion measure
            // (otherwise each chart's arbitrary ABF++ scale inflates the variance).
            double sumA2 = 0.0;
            for (index_t sf = 0; sf < sub.facets.nb(); sf++) {
                index_t a = sub.facets.vertex(sf, 0);
                index_t b = sub.facets.vertex(sf, 1);
                index_t d = sub.facets.vertex(sf, 2);
                double ux0 = uv[2*a], uy0 = uv[2*a+1];
                double ux1 = uv[2*b], uy1 = uv[2*b+1];
                double ux2 = uv[2*d], uy2 = uv[2*d+1];
                sumA2 += 0.5 * std::abs((ux1-ux0)*(uy2-uy0) - (ux2-ux0)*(uy1-uy0));
            }
            double sumA3 = 0.0;
            for (size_t i = 0; i < cf.size(); i++) {
                index_t f = cf[i];
                auto p0 = mesh.vertices.point(mesh.facets.vertex(f, 0));
                auto p1 = mesh.vertices.point(mesh.facets.vertex(f, 1));
                auto p2 = mesh.vertices.point(mesh.facets.vertex(f, 2));
                double cx = (p1[1]-p0[1])*(p2[2]-p0[2]) - (p1[2]-p0[2])*(p2[1]-p0[1]);
                double cy = (p1[2]-p0[2])*(p2[0]-p0[0]) - (p1[0]-p0[0])*(p2[2]-p0[2]);
                double cz = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0]);
                sumA3 += 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
            }
            double sc = (sumA2 > 1e-18) ? std::sqrt(sumA3 / sumA2) : 1.0;
            for (size_t i = 0; i < origv.size(); i++) {
                chart_uv[c][origv[i]] = std::make_pair(uv[2 * i] * sc, uv[2 * i + 1] * sc);
            }
        }
    }

    // Translate-pack charts into a grid (NO per-chart rescale: each chart is
    // already area-normalized). Then globally normalize the atlas to [0,1]^2
    // (uniform global scale does not affect per-triangle area ratios).
    {
        index_t cols = (index_t)std::ceil(std::sqrt((double)std::max(ncharts, (index_t)1)));
        index_t rows = (ncharts + cols - 1) / cols;
        if (rows < 1) rows = 1;
        double cw = 1.0 / cols, ch = 1.0 / rows;
        for (index_t c = 0; c < ncharts; c++) {
            if (chart_uv[c].empty()) continue;
            double umin = 1e18, vmin = 1e18;
            for (std::map<index_t, std::pair<double, double>>::iterator it = chart_uv[c].begin();
                 it != chart_uv[c].end(); ++it) {
                umin = std::min(umin, it->second.first);
                vmin = std::min(vmin, it->second.second);
            }
            index_t cx = c % cols, cy = c / cols;
            double ox = cx * cw - umin + 0.05 * cw;
            double oy = cy * ch - vmin + 0.05 * ch;
            for (std::map<index_t, std::pair<double, double>>::iterator it = chart_uv[c].begin();
                 it != chart_uv[c].end(); ++it) {
                it->second.first += ox;
                it->second.second += oy;
            }
        }
        double gminu = 1e18, gmaxu = -1e18, gminv = 1e18, gmaxv = -1e18;
        for (index_t c = 0; c < ncharts; c++) {
            for (std::map<index_t, std::pair<double, double>>::iterator it = chart_uv[c].begin();
                 it != chart_uv[c].end(); ++it) {
                gminu = std::min(gminu, it->second.first);  gmaxu = std::max(gmaxu, it->second.first);
                gminv = std::min(gminv, it->second.second); gmaxv = std::max(gmaxv, it->second.second);
            }
        }
        double du = std::max(gmaxu - gminu, 1e-12);
        double dv = std::max(gmaxv - gminv, 1e-12);
        double gs = 1.0 / std::max(du, dv);
        for (index_t c = 0; c < ncharts; c++) {
            for (std::map<index_t, std::pair<double, double>>::iterator it = chart_uv[c].begin();
                 it != chart_uv[c].end(); ++it) {
                it->second.first = (it->second.first - gminu) * gs;
                it->second.second = (it->second.second - gminv) * gs;
            }
        }
    }

    // Write uv.obj: dedup vt by (chart, vertex) so adjacent triangles connect.
    std::ofstream out(out_path);
    if (!out) { std::cerr << "Error: cannot write " << out_path << "\n"; return 4; }
    out.precision(17);
    for (index_t v = 0; v < mesh.vertices.nb(); v++) {
        auto p = mesh.vertices.point(v);
        out << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
    }
    std::map<std::pair<index_t, index_t>, index_t> vt_map;
    std::vector<std::string> vt_lines, face_lines;
    face_lines.reserve(mesh.facets.nb());
    for (index_t f = 0; f < mesh.facets.nb(); f++) {
        index_t ch = chart[f];
        std::ostringstream fl; fl << "f";
        index_t nbf = mesh.facets.nb_vertices(f);
        for (index_t lv = 0; lv < nbf; lv++) {
            index_t ov = mesh.facets.vertex(f, lv);
            std::pair<index_t, index_t> key(ch, ov);
            std::map<std::pair<index_t, index_t>, index_t>::iterator it = vt_map.find(key);
            index_t vtidx;
            if (it == vt_map.end()) {
                vtidx = vt_lines.size() + 1;
                double u = 0.0, vv = 0.0;
                std::map<index_t, std::pair<double, double>>::iterator jit = chart_uv[ch].find(ov);
                if (jit != chart_uv[ch].end()) { u = jit->second.first; vv = jit->second.second; }
                std::ostringstream s; s.precision(17); s << "vt " << u << " " << vv;
                vt_lines.push_back(s.str());
                vt_map[key] = vtidx;
            } else {
                vtidx = it->second;
            }
            fl << " " << (ov + 1) << "/" << vtidx;
        }
        face_lines.push_back(fl.str());
    }
    for (size_t i = 0; i < vt_lines.size(); i++) out << vt_lines[i] << "\n";
    for (size_t i = 0; i < face_lines.size(); i++) out << face_lines[i] << "\n";

    std::cerr << "abf_unwrap OK: V=" << mesh.vertices.nb()
              << " F=" << mesh.facets.nb()
              << " seam_edges=" << seams.size()
              << " charts=" << ncharts
              << " abf_ok=" << abf_ok << " abf_fail=" << abf_fail
              << " -> " << out_path << "\n";
    return 0;
}
