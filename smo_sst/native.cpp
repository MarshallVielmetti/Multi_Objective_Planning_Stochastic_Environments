// Fused CPU particle transitions; no Python work inside a particle/time loop.
// clang++/g++ -O3 -std=c++17 -shared -fPIC native.cpp -o kernel.so
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>

namespace {
// Persistent workers amortize thread creation over all planner edges.
// The calling thread participates; RNG states belong to particles, not workers.
class Pool {
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable ready, done;
    std::function<void(int,int)> job;
    int generation=0, pending=0, count=0;
    bool stop=false;
public:
    const int size;
    explicit Pool(int n): size(n) {
        for(int id=1;id<n;++id) workers.emplace_back([this,id] {
            int seen=0;
            std::unique_lock<std::mutex> lock(mutex);
            for(;;) {
                ready.wait(lock,[&]{return stop || generation!=seen;});
                if(stop) return;
                seen=generation;
                int lo=count*id/size,hi=count*(id+1)/size;
                lock.unlock();job(lo,hi);lock.lock();
                if(--pending==0) done.notify_one();
            }
        });
    }
    ~Pool() {
        {std::lock_guard<std::mutex> lock(mutex);stop=true;}
        ready.notify_all();
        for(auto& t:workers)t.join();
    }
    void run(int n, std::function<void(int,int)> work) {
        if(size==1 || n<32) {work(0,n);return;}
        {std::lock_guard<std::mutex> lock(mutex);
            count=n;job=std::move(work);pending=size-1;++generation;}
        ready.notify_all();job(0,n/size);
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock,[&]{return pending==0;});
    }
};
std::mutex dispatch_mutex;
std::unique_ptr<Pool> pool;
inline double clip(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }
// SplitMix64, with one independent state per particle. No global RNG state.
inline uint64_t next(uint64_t& state) {
    uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
inline double uniform(uint64_t& state) { return (next(state) >> 11) * 0x1.0p-53; }
// Exact rejection sampler for 2*Beta(3,3)-1; density proportional to (1-x*x)^2.
inline double beta_noise(uint64_t& state) {
    for (;;) {
        double x = 2 * uniform(state) - 1, v = 1 - x*x;
        if (uniform(state) < v*v) return x;
    }
}
inline double ground(const float* a, const float* b, int A, double D, double C) {
    double ego = 0, env = 0;
    bool modes = false;
    for (int j=0; j<4; ++j) { double d=double(a[j])-b[j]; ego+=d*d; }
    for (int i=0; i<A; ++i) {
        for (int j=0; j<3; ++j) { double d=double(a[4+4*i+j])-b[4+4*i+j]; env+=d*d; }
        modes |= (a[7+4*i] != b[7+4*i]);
    }
    return (std::sqrt(ego)+std::sqrt(env)+C*modes)/D + std::abs(a[4+4*A]-b[4+4*A]);
}

inline double survival_at(const double* table, int L, double age, double steps) {
    // Bilinear interpolation preserves continuity at both age and time knots.
    age=clip(age,0,L);steps=clip(steps,0,L);
    int n=int(age),k=int(steps),n1=std::min(n+1,L),k1=std::min(k+1,L);
    double f=steps-k,g=age-n;
    double a=(1-f)*table[n*(L+1)+k]+f*table[n*(L+1)+k1];
    double b=(1-f)*table[n1*(L+1)+k]+f*table[n1*(L+1)+k1];
    return (1-g)*a+g*b;
}

inline void control(const float* s, int A, double qx, double qy, const double* c,
                    const double* survival, double radial, double tangential,
                    double sn, double cs, double& acc, double& omega) {
    double x=s[0],y=s[1],v=s[3],eps2=c[12]*c[12];
    double dx=qx-x,dy=qy-y,norm=std::sqrt(dx*dx+dy*dy+eps2);
    dx/=norm;dy/=norm;
    if(c[17]==2 && (radial!=0 || tangential!=0)) {
        double rx=0,ry=0,ws=0;
        for(int i=0;i<A;++i) if(s[7+4*i]==1) {
            double ex=x-s[4+4*i],ey=y-s[5+4*i];
            double den=std::sqrt(ex*ex+ey*ey+eps2);
            double ux=c[4]*ex/den-v*cs,uy=c[4]*ey/den-v*sn;
            double t=clip((ex*ux+ey*uy)/(ux*ux+uy*uy+c[20]*c[20]),0,c[18]);
            double mx=-ex+t*ux,my=-ey+t*uy;
            double near=1/(1+(mx*mx+my*my)/(c[19]*c[19]));
            double w=survival_at(survival,int(c[9]),s[6+4*i],t/c[0])*near*near/(1+t/c[21]);
            rx+=w*ex/den;ry+=w*ey/den;ws+=w;
        }
        // R_{pi/2}(rx,ry)=(-ry,rx). Signed tangential gain sets the passing side.
        dx+=(radial*rx-tangential*ry)/(1+ws);
        dy+=(radial*ry+tangential*rx)/(1+ws);
    } else if(c[17]==1 && c[11]!=0) {
        // Preserve the original fixed-gain, distance-only reactive controller.
        double rx=0,ry=0,ws=0;
        for(int i=0;i<A;++i) if(s[7+4*i]==1) {
            double ex=x-s[4+4*i],ey=y-s[5+4*i],d2=ex*ex+ey*ey;
            if(d2<c[10]*c[10]) {
                double w=clip((c[10]-std::sqrt(d2))/(c[10]-c[8]),0,1);
                double den=std::sqrt(d2+eps2);
                rx+=w*ex/den;ry+=w*ey/den;ws+=w;
            }
        }
        dx+=c[11]*rx/(1+ws);dy+=c[11]*ry/(1+ws);
    }
    norm=std::sqrt(dx*dx+dy*dy+eps2);dx/=norm;dy/=norm;
    omega=c[3]*std::tanh(3*(-sn*dx+cs*dy));
    acc=c[2]*std::tanh(2*(c[1]-v)/c[1]);
}
}

extern "C" {
void set_threads(int n) {
    std::lock_guard<std::mutex> lock(dispatch_mutex);
    if(!pool || pool->size!=n) pool=std::make_unique<Pool>(std::max(1,n));
}
void noise_samples(uint64_t seed, double* out, int n) {
    for(int i=0;i<n;++i) out[i]=beta_noise(seed);
}

// c: dt,vmax,amax,omax,adv_speed,pfire,rmin,rmax,rcap,L,Rrep,krep,eps,
//    noise_xy,noise_theta,noise_v,noise_adv,family,H,guard_radius,velocity_eps,urgency_time
void controls(const float* cloud,int N,int A,double qx,double qy,const double* c,
              const double* survival,double radial,double tangential,double* out) {
    int dim=6+4*A;
    for(int p=0;p<N;++p) {
        const float* s=cloud+p*dim;
        control(s,A,qx,qy,c,survival,radial,tangential,std::sin(double(s[2])),std::cos(double(s[2])),
                out[2*p],out[2*p+1]);
    }
}

void propagate(const float* input, float* output, uint64_t* states,
               int N, int A, int steps, double qx, double qy, const double* c,
               const double* survival,double radial,double tangential) {
    const int dim=6+4*A, bi=4+4*A, ci=bi+1;
    const double dt=c[0], vmax=c[1];
    std::memcpy(output,input,sizeof(float)*N*dim);
    std::lock_guard<std::mutex> lock(dispatch_mutex);
    if(!pool) pool=std::make_unique<Pool>(1);
    pool->run(N,[&](int begin,int end) {
    for(int p=begin;p<end;++p) {
        float* s=output+p*dim;
        uint64_t rng=states[p];
        for(int t=0;t<steps;++t) {
            const double x=s[0],y=s[1],theta=s[2],v=s[3];
            double sn=std::sin(theta),cs=std::cos(theta);
            double acc,omega;
            control(s,A,qx,qy,c,survival,radial,tangential,sn,cs,acc,omega);
            s[ci]+=dt*(1+0.05*((acc/c[2])*(acc/c[2])+(omega/c[3])*(omega/c[3])));
            s[0]=x+v*cs*dt+c[13]*beta_noise(rng);
            s[1]=y+v*sn*dt+c[13]*beta_noise(rng);
            s[2]=theta+omega*dt+c[14]*beta_noise(rng);
            s[3]=clip(v+acc*dt+c[15]*beta_noise(rng),-vmax,vmax);
            for(int i=0;i<A;++i) {
                float* z=s+4+4*i;
                double ex=x-z[0],ey=y-z[1],d2=ex*ex+ey*ey;
                if(z[3]==0) {
                    double fire=c[5]*clip((c[7]*c[7]-d2)/(c[7]*c[7]-c[6]*c[6]),0,1);
                    if(fire>0 && uniform(rng)<fire) z[3]=1;
                } else if(z[3]==1) {
                    // Transition hazard is evaluated at n_t, pursuit uses x_t.
                    double hazard=clip((z[2]-0.8*(c[9]-1))/(0.2*(c[9]-1)),0,1);
                    bool terminal=uniform(rng)<hazard;
                    double d=std::sqrt(d2);
                    double scale=d>1e-12 ? c[4]*dt/d : 0;
                    z[0]+=scale*ex+c[16]*beta_noise(rng);
                    z[1]+=scale*ey+c[16]*beta_noise(rng);
                    z[2]+=1;
                    if(terminal) z[3]=2;
                }
                // New activations can capture, new terminal projectiles cannot.
                double cx=s[0]-z[0],cy=s[1]-z[1];
                if(z[3]==1 && cx*cx+cy*cy<=c[8]*c[8]) s[bi]=1;
            }
        }
        states[p]=rng;
    }
    });
}

void coupling_costs(const float* a, const float* b, const int64_t* ao,
                    const int64_t* bo, int N, int A, int P, double D, double C, double* out) {
    int dim=6+4*A;
    for(int j=0;j<P;++j) {
        double total=0;
        for(int i=0;i<N;++i) total+=ground(a+ao[j*N+i]*dim,b+bo[j*N+i]*dim,A,D,C);
        out[j]=total/N;
    }
}
void cost_matrix(const float* a,const float* b,int N,int A,double D,double C,double* out) {
    int dim=6+4*A;
    for(int i=0;i<N;++i) for(int j=0;j<N;++j) out[i*N+j]=ground(a+i*dim,b+j*dim,A,D,C);
}
}
