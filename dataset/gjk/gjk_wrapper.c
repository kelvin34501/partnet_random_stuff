#include "gjk.h"
//#include <stdio.h>

static int
polyhedron_support(double *support, const double *d,
    const double *verts, int cnt)
{
    int imax = 0;
    double dmax = f3dot(verts, d);
    for (int i = 1; i < cnt; ++i) {
        /* find vertex with max dot product in direction d */
        double dot = f3dot(&verts[i*3], d);
        if (dot < dmax) continue;
        imax = i, dmax = dot;
    } f3cpy(support, &verts[imax*3]);
    return imax;
}

extern double
polyhedron_intersect_polyhedron(
    const double *averts, int acnt,
    const double *bverts, int bcnt)
{
    /* initial guess */
    double d[3] = {0};
    struct gjk_support s = {0};
    f3cpy(s.a, averts);
    f3cpy(s.b, bverts);
    f3sub(d, s.b, s.a);
    //printf("d: %f %f %f\n", d[0], d[1], d[2]);
    //printf("averts: %f %f %f\n", averts[0], averts[1], averts[2]);
    //printf("bverts: %f %f %f\n", bverts[0], bverts[1], bverts[2]);

    /* run gjk algorithm */
    struct gjk_simplex gsx = {0};
    while (gjk(&gsx, &s, d)) {
        /* transform direction */
        double n[3]; f3mul(n, d, -1);
        //printf("d: %f %f %f\n", d[0], d[1], d[2]);
        //printf("n: %f %f %f\n", n[0], n[1], n[2]);

        /* run support function on tranformed directions  */
        s.aid = polyhedron_support(s.a, n, averts, acnt);
        s.bid = polyhedron_support(s.b, d, bverts, bcnt);

        /* calculate distance vector on transformed points */
        f3sub(d, s.b, s.a);
    }
    /* check distance between closest points */
    struct gjk_result res;
    //printf("%d\n", gsx.hit);
    gjk_analyze(&res, &gsx);
    //printf("%f\n", res.distance_squared);
    return res.distance_squared;
}
