#ifndef GJK_H
#define GJK_H

#define GJK_MAX_ITERATIONS 20
#define GJK_FLT_MAX 3.40282347E+38F
#define GJK_EPSILON 1.19209290E-07F

static const double f3z[3] = {0};
#define fop(r,e,a,p,b,i,s) (r) e ((a) p (b)) i (s)
#define f3op(r,e,a,p,b,i,s) do {\
    fop((r)[0],e,(a)[0],p,(b)[0],i,s),\
    fop((r)[1],e,(a)[1],p,(b)[1],i,s),\
    fop((r)[2],e,(a)[2],p,(b)[2],i,s);}while(0)
#define f3cpy(d,s) (d)[0]=(s)[0],(d)[1]=(s)[1],(d)[2]=(s)[2]
#define f3add(d,a,b) f3op(d,=,a,+,b,+,0)
#define f3sub(d,a,b) f3op(d,=,a,-,b,+,0)
#define f3mul(d,a,s) f3op(d,=,a,+,f3z,*,s)
#define f3dot(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define f3cross(d,a,b) do {\
    (d)[0] = ((a)[1]*(b)[2]) - ((a)[2]*(b)[1]),\
    (d)[1] = ((a)[2]*(b)[0]) - ((a)[0]*(b)[2]),\
    (d)[2] = ((a)[0]*(b)[1]) - ((a)[1]*(b)[0]);}while(0)

struct gjk_support {
    int aid, bid;
    double a[3];
    double b[3];
};
struct gjk_simplex {
    int max_iter, iter;
    int hit, cnt;
    struct gjk_vertex {
        double a[3];
        double b[3];
        double p[3];
        int aid, bid;
    } v[4];
    double bc[4], D;
};
struct gjk_result {
    int hit;
    double p0[3];
    double p1[3];
    double distance_squared;
    int iterations;
};
extern int gjk(struct gjk_simplex *s, const struct gjk_support *sup, double *dv);
extern void gjk_analyze(struct gjk_result *res, const struct gjk_simplex *s);
extern void gjk_quad(struct gjk_result *res, double a_radius, double b_radius);

#endif
