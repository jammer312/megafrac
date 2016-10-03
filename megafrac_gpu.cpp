#include <iostream>
#include <fstream>
#include <SFML/Graphics.hpp>
#include <vector>
#include <algorithm>
#include <string>

#include <CL/cl2.hpp>

#define SCREEN_X 600
#define SCREEN_Y SCREEN_X

//----------//
#define cur_iterations 255
//----------//

using namespace std;

class complex
{
	private:
		complex power_pos(int power,complex& to_power)
		{
			if(power<=0)
				return complex(1,0);
			return to_power*power_pos(power-1,to_power);
		}
	public:
		double real;
		double imaginary;
		complex(double r=0,double im=0): real(r),imaginary(im){};
		complex in_power(int power)
		{
			return power_pos(power,*this);
		}
		double module_squared()
		{
			return real*real+imaginary*imaginary;
		}
		bool valid_for_frac()
		{
			return module_squared()<4;
		}
		complex scalar_multiply(double a)
		{
			return complex(real*a,imaginary*a);
		}
		complex operator+(complex a)
		{
			return complex(real+a.real,imaginary+a.imaginary);
		}
		complex operator-(complex a)
		{
			return complex(real-a.real,imaginary-a.imaginary);
		}
		complex operator*(complex a)
		{
			return complex(real*a.real-imaginary*a.imaginary,real*a.imaginary+imaginary*a.real);
		}
};

cl::Program program;
cl::Context context;
cl::CommandQueue queue;
cl::Kernel test;

complex UL_point(-2,-2),LR_point(2,2);
complex from_pixel_to_point(sf::Vector2u pixel,int resX=SCREEN_X,int resY=SCREEN_Y)
{
	complex delta=(LR_point-UL_point);
	return (UL_point+complex(delta.real*pixel.x*1.0/resX,delta.imaginary*pixel.y*1.0/resY));
}
cl_double2 make_cl_double2(double a,double b)
{
	cl_double2 tmp;
	tmp.s[0]=a;
	tmp.s[1]=b;
	return tmp;
}
sf::Color palette[255];
unsigned char result[600][600];
complex constant(0,0);
bool mandel=true;
int power=2;
void calculate(sf::Image* to)
{
	//sf::Clock clock;
	cl::Buffer clmOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,600*600*(sizeof(char)),result);
	test.setArg(0,clmOutputVector);
	test.setArg(1,make_cl_double2(UL_point.real,UL_point.imaginary));
	test.setArg(2,make_cl_double2(LR_point.real,LR_point.imaginary));
	test.setArg(3,make_cl_double2(constant.real,constant.imaginary));
	test.setArg(4,(int)mandel);
	test.setArg(5,power);
	queue.enqueueNDRangeKernel(test,cl::NullRange,cl::NDRange(600,600));
	queue.finish();
	queue.enqueueReadBuffer(clmOutputVector, CL_TRUE, 0,600*600,result);
	to->create(600,600);
	for(int i=0;i<600;++i)
		for(int l=0;l<600;++l)
			to->setPixel(l,i,palette[result[i][l]]);
	//cout<<"Calculations taken "<<clock.getElapsedTime().asSeconds()<<" seconds\n";
}

void load_palette(std::string filename)
{
	//generate_palette();//for safety purposes
	sf::Image palette_img;
	palette_img.loadFromFile(filename);
	int size=palette_img.getSize().x;
	for(int i=0;i<size&&i<cur_iterations;i++)
	{
		palette[i]=palette_img.getPixel(i,0);
	}
}
void smart_gradient(complex from,complex to,int frames,std::string save_dir)
{
	cout<<"Calculating gradient from "<<from.real<<' '<<from.imaginary<<" to "<<to.real<<' '<<to.imaginary<<endl;
	complex backup_constant=constant;
	complex delta=to-from;
	sf::Image tmp;
	tmp.create(SCREEN_X,SCREEN_Y);
	std::cout<<"Saving gradient to ./saves/"<<save_dir<<'\n';
	{
		std::string tmp="mkdir saves/";
		std::system((tmp+save_dir).c_str());
		std::ofstream ind("./saves/"+save_dir+"/index");
		ind<<frames;
		ind.close();
	}
	for(int i=0;i<frames;++i)
	{
		//antiheat
		//if(i%25==1)
		//	std::system("sleep 60");
		//
		constant=complex((delta.real*i*1.0)/frames+from.real,delta.imaginary*(i*1.0/frames)+from.imaginary);
		std::cout<<i<<'/'<<frames<<'\r';
		std::cout.flush();
		calculate(&tmp);
		tmp.saveToFile("./saves/"+save_dir+'/'+std::to_string(i)+".png");
	}
	std::cout<<frames<<'/'<<frames<<".Done.\n";
	constant=backup_constant;
	//std::system("shutdown now");
}
int main(int arga,char* args[])
{
	//------------------------------//
	//OpenCL init//
	//------------------------------//
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	std::vector<cl::Device> devices;
	platforms[0].getDevices(CL_DEVICE_TYPE_GPU,&devices);
	cl::Device gpu=devices[0];
	cout<<"Using GPU \""<<gpu.getInfo<CL_DEVICE_NAME>()<<"\""<<endl;
	vector<cl::Device> context_devices;
	context_devices.push_back(gpu);
	context=cl::Context(context_devices);
	std::ifstream sourceFile("kernel_megafrac_gpu.cl");
	std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),(std::istreambuf_iterator<char>()));
	cl::Program::Sources src;
	src.push_back(sourceCode);
	program=cl::Program(context,src);
	program.build(context_devices);
	test=cl::Kernel(program,"main");
	queue=cl::CommandQueue(context, gpu);
	//------------------------------//
	//End of OpenCL init//
	//------------------------------//
	load_palette("palette_for_fractal.png");
	//gradient from from to to frames saveto con con type jmp jmp
	if(arga>=2)
	{
		if((string)args[1]=="gradient")
		{
			if(arga<13)
			{
				cout<<"Usage: "<<args[0]<<" gradient from{x,y} to{x,y} frames saveto type(julia?) jmp{x1,y1,x2,y2}"<<endl;
				return 0;
			}
			complex from(stold(args[2]),stold(args[3]));
			complex to(stold(args[4]),stold(args[5]));
			int frames=stoi(args[6]);
			string output_folder=args[7];
			mandel=!stoi(args[8]);
			UL_point=complex(stold(args[9]),stold(args[10]));
			LR_point=complex(stold(args[11]),stold(args[12]));
			if(arga>=14)
				power=stoi(args[13]);
			smart_gradient(from,to,frames,output_folder);
			return 0;
		}
	}
	if(arga>=3)
	{
		constant=complex(std::stold(args[1]),std::stold(args[2]));
	}
	if(arga>=4)
	{
		if(string(args[3])=="julia"||string(args[3])=="1")
			mandel=false;
	}
	if(arga>=8)
	{
		UL_point=complex(stold(args[4]),stold(args[5]));
		LR_point=complex(stold(args[6]),stold(args[7]));	
	}
	if(arga>=9)
		power=stoi(args[8]);
	//------------------------------//
	sf::RenderWindow window(sf::VideoMode(SCREEN_X,SCREEN_Y),"GPU_RULEZ",sf::Style::Titlebar|sf::Style::Close);
	bool resize_event=false;
	sf::Vector2u firstPoint;
	sf::Image drawImg;
	drawImg.create(600,600);
	sf::Texture drawText;
	bool redraw=true;
	calculate(&drawImg);
	while(window.isOpen())
	{
		sf::Event event;
		while(window.pollEvent(event))
		{
			if(event.type==sf::Event::Closed)
				window.close();
			if(event.type==sf::Event::MouseButtonPressed&&event.mouseButton.button==sf::Mouse::Left)
			{
				resize_event=true;
				firstPoint=sf::Vector2u(event.mouseButton.x,event.mouseButton.y);
			}
			if(event.type==sf::Event::MouseButtonReleased&&event.mouseButton.button==sf::Mouse::Left)
			{
				resize_event=false;
				complex new_UL_point=from_pixel_to_point(firstPoint);
				uint length = std::max(event.mouseButton.x-firstPoint.x,event.mouseButton.y-firstPoint.y);
				complex new_LR_point=from_pixel_to_point(firstPoint+sf::Vector2u(length,length));
				UL_point=new_UL_point;
				LR_point=new_LR_point;
				std::cout<<"***STAND BY***UPDATING***\n";
				calculate(&drawImg);
				std::cout<<"***DONE***\n";
				redraw=true;
			}
			if(event.type==sf::Event::KeyReleased)
			{
				if(event.key.code==sf::Keyboard::Space)
				{
					drawImg.saveToFile("./saves/tmp_save.png");
					cout<<"Image saved to ./saves/tmp_save.png"<<endl;
				}
			}
		}
		if(resize_event)
		{
			window.clear();
			sf::Sprite drawable(drawText);
			window.draw(drawable);
			sf::VertexArray vertices;
			vertices.setPrimitiveType(sf::LinesStrip);
			sf::Vertex vertex;
			vertex.color=sf::Color::Red;
			vertex.position=sf::Vector2f(firstPoint.x,firstPoint.y);
			vertices.append(vertex);
			uint length=std::max(sf::Mouse::getPosition(window).x-firstPoint.x,sf::Mouse::getPosition(window).y-firstPoint.y);
			vertex.position=sf::Vector2f(firstPoint.x+length,firstPoint.y);
			vertices.append(vertex);
			vertex.position=sf::Vector2f(firstPoint.x+length,firstPoint.y+length);
			vertices.append(vertex);
			vertex.position=sf::Vector2f(firstPoint.x,firstPoint.y+length);
			vertices.append(vertex);
			vertex.position=sf::Vector2f(firstPoint.x,firstPoint.y);
			vertices.append(vertex);
			window.draw(vertices);
			window.display();
		}
		else if(redraw)
		{
			window.clear();
			drawText.loadFromImage(drawImg);
			sf::Sprite drawable(drawText);
			window.draw(drawable);
			window.display();
			redraw=false;
		}
	}
}