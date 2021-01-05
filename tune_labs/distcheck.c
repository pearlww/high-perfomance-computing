#include "data.h"

#ifdef ARRAY_OF_STRUCTS
double 
distcheck(particle_t *p, int n) {

    double dist = 0;
    double d;
    for (int i=0;i<n;i++){
        d = sqrt(p[i].x*p[i].x + p[i].y*p[i].y+p[i].z*p[i].z);
        p[i].dist = d;
        dist+=d;
    }

    return dist;
}
#else
double 
distcheck(particle_t p, int n) {

    double dist = 0;
    double d;
    for (int i=0;i<n;i++){
        d = sqrt(p.x[i]*p.x[i] + p.x[i]*p.x[i] +p.x[i]*p.x[i] );
        dist+=d;
    }
    return dist;
}
#endif
