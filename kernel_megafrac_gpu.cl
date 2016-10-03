#pragma OPENCL_EXTENSION cl_khr_fp64 : enable
#define SCREEN_X 600
#define SCREEN_Y 600

typedef double2 Complex;

Complex multiply(Complex a, Complex b)
{
	return (Complex)(a.s0*b.s0-a.s1*b.s1, a.s1*b.s0+a.s0*b.s1);
}
Complex comp_pow(Complex a,int power)
{
	Complex tmp=a;
	for(int i=1;i<power;++i)
	{
		tmp=multiply(tmp,a);
	}
	return tmp;
}
Complex multiply_scalar(Complex a,double b)
{
	return (Complex)(a.s0*b,a.s1*b);
}
Complex delta(Complex a,Complex b)
{
	return (Complex)(b.s0-a.s0,b.s1-a.s1);
}
Complex sum(Complex a,Complex b)
{
	return (Complex)(b.s0+a.s0,b.s1+a.s1);
}
float normalized_iterations(int n, Complex zn, int bailout)
{
	return n + (log(log(convert_float(bailout)))-log(log(length(zn))))/log(2.0);
}

float boundedorbit(Complex seed, Complex c, float bound, int bailout,int power)
{
	Complex z = comp_pow(seed,power) + c;
	for (int k = 0; k < bailout; k++) {
		if (length(z) > bound) 
			return normalized_iterations(k, z, bailout);
		z = comp_pow(z,power) + c;
	}
	return FLT_MIN;
}

unsigned char grayvalue(float n) {
	return convert_uchar_sat_rte(n);
}

__kernel void main(__global unsigned char* output,double2 f,double2 t,double2 cnst,int mandel,int power)
{
	int k = get_global_id(0);
	int j = get_global_id(1);
	Complex c=(Complex)(f.s0+(t.s0-f.s0)*(k*1.0/SCREEN_X),f.s1+(t.s1-f.s1)*(j*1.0/SCREEN_Y));
	float count = boundedorbit(multiply_scalar(c,1-mandel%2)+multiply_scalar(cnst,mandel),multiply_scalar(c,mandel)+multiply_scalar(cnst,1-mandel%2), 2.0, 255, power);
	output[SCREEN_Y*j+k]=grayvalue(count);
}