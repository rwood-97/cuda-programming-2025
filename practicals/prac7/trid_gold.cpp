
void gold_trid(int NX, int niter, float* u, float* c)
{
  float lambda=1.0f, bb, cc, dd;
  float a_i;
  float	d[NX];

  for (int iter=0; iter<niter; iter++) {

    //
    // forward pass
    //

    bb   =  2.0f + lambda;
    cc   = -1.0f;
    dd   = lambda*u[0];

    c[0]   = cc/bb;
    d[0]   = dd/bb;
    u[0] = d[0];

    for (int i=1; i<NX; i++) {
      a_i = -1.0f;
      bb   = 2.0f + lambda - a_i*c[i-1];
      dd   = lambda*u[i] - a_i*u[i-1];
      c[i]  = -1.0f/bb;
      d[i] = dd/bb;
      u[i] = d[i];
    }

    //
    // reverse pass
    //

    u[NX-1] = d[NX-1];

    for (int i=NX-2; i>=0; i--) {
      u[i]   = u[i] - c[i]*u[i+1];
    }
  }
}


