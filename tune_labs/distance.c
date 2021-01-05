#include "data.h"

#ifdef ARRAY_OF_STRUCTS

double 
distance(particle_t *p, int n) {
    
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
distance(particle_t p, int n) {

    double dist = 0;
    double d;
    for (int i=0;i<n;i++){
        d = sqrt(p.x[i]*p.x[i] + p.y[i]*p.y[i] +p.z[i]*p.z[i] );
        p.dist[i]=d;
        dist+=d;
    }
    return dist;
}
#endif
