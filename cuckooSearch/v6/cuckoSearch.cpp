//Cuckoo search simple test implementation travel salesman problem

#include <iostream>
#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <algorithm>    // std::random_shuffle
#include <math.h>       /* pow */
#include <random>		/* normal distribution */
#include "cec14_test_func.cpp"
#include <omp.h>

#define PI 3.1415926535897932384626433832795029

void cec14_test_func(double *, double *,int,int,int);

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;

int i,j,k,n,m,func_num;
double *f,*x;

using namespace std;









class CuckooSearch{
private:
	int numeroNidos; //Cada nido es una solucion
	float probabilidadDescubrimiento;
	
	//float tolerancia;
	int nd; //Dimension de la solucion 

	//vector<int> lowerBound; 
	//vector<int> upperBound; 

	int nLowerBound,nUpperBound;
	int maxEvals;

	float lambda; //Variable que controla la distribución levy

	float epsilon; //Variable que controla la local search

	bool maximizacion; //Controla si vamos a maximizar o a minimizar la funcion objetivo

	//FuncionesObjetivo funciones; //Clase que contiene las funciones objetivo.

	float probReinicializacion; //Probabilidad de reinicializacion en la version 4

	int nAgentes; //Controla el número de agentes que buscan soluciones en la version 5
	
	int maxEvalsBL;

	double aleatorio(double lowBound, double highBound){
		return (lowBound + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(highBound-lowBound))));
	}

	vector<double> generateRandomSolution(){
		vector<double> s(nd);
		for(int i=0;i<s.size();i++){
			s[i]=aleatorio(nLowerBound,nUpperBound);
		}
		return s;
	}


	//vector <double> levyFlight(vector<double> puntoPartida){
		/*vector<double> nuevoNido(puntoPartida.size());
		
		while (nuevoNido[0]<nLowerBound || nuevoNido[0]>nUpperBound || nuevoNido[1]<nLowerBound || nuevoNido[1]>nUpperBound){
			double theta=aleatorio(0,1)*2*PI;
			double f=pow(aleatorio(0,1),(-1/lambda));
			nuevoNido[0]=puntoPartida[0]+f*cos(theta);
			nuevoNido[1]=puntoPartida[1]+f*sin(theta);
		}
		
		return nuevoNido;*/

	//}

	
	vector<double> levyFlight(vector<double> puntoPartida, vector<double> mejorSolucion){
		double beta = 3/2;
		double sigma = pow((tgamma(1.0+beta) * sin(PI*beta/2.0) / ((tgamma(1.0+beta)/2.0) * beta * (pow(2.0,((beta-1.0)/2.0))))),(1.0/beta));

		vector<double> nu=puntoPartida; //nu=x a priori
		vector<double> solucionFinal=puntoPartida;

		default_random_engine generator;
		normal_distribution<double> distributionU(0.0,pow(sigma,2));
		normal_distribution<double> distributionV(0.0,1);

		double u=distributionU(generator);
		double v=distributionV(generator);

		//double step = u / pow(fabs(v),(1/beta));
		double step = u / pow(fabs(v),(-lambda));

		//nu=epsilon*step*(x-xbest)
		for(int i=0;i<nu.size();i++){
			nu[i]-=mejorSolucion[i];
		}

		double stepsize = epsilon*step;

		for(int i=0;i<nu.size();i++){
			nu[i]*=stepsize;
		}

		vector<double> nuUpsilon(nu.size());
		for(int i=0;i<nuUpsilon.size();i++){
			nuUpsilon[i]=nu[i]*distributionV(generator);
			solucionFinal[i]+=nuUpsilon[i];
		}


		return solucionFinal;

		/*double beta = 3/2;
		double sigma = pow((tgamma(1.0+beta) * sin(PI*beta/2.0) / ((tgamma(1.0+beta)/2.0) * beta * (pow(2.0,((beta-1.0)/2.0))))),(1.0/beta));

		bool correcto=false;

		vector<double> xRestaXbest=puntoPartida;
		vector<double> solucionFinal=puntoPartida;

		while(!correcto){
			correcto=true;
			default_random_engine generator;
			normal_distribution<double> distributionU(0.0,sigma);
			normal_distribution<double> distributionV(0.0,1);

			double u=distributionU(generator);
			double v=distributionV(generator);

			double step = u / pow(fabs(v),(1/beta));

			for(int i=0;i<xRestaXbest.size();i++){
				xRestaXbest[i]-=mejorSolucion[i];
			}

			double stepsize = 0.01 * step;

			for(int i=0;i<xRestaXbest.size();i++){
				xRestaXbest[i]*=stepsize;
			}

			vector<double> nuPorGammaRandom(solucionFinal.size());

			for(int i=0;i<solucionFinal.size();i++){
				nuPorGammaRandom[i]=xRestaXbest[i]*distributionV(generator);
				solucionFinal[i]+=nuPorGammaRandom[i];
				if(solucionFinal[i]>nUpperBound || solucionFinal[i]<nLowerBound){correcto=false;}
			}



		}	

		return solucionFinal;*/
	}


	void localSearch(int nFunction,vector<double>& solucion, double& fitnessSolucion, int& contEvals, int maxEvalsBL){
		bool huboMejora=true;
		int contEvalsBL=0;
		vector<double> vecinoActual=solucion;
		double fitnessVecinoActual=fitnessSolucion;

		while(huboMejora && contEvals<maxEvals && contEvalsBL<maxEvalsBL){
			huboMejora=false;
			for(int iter=0;iter<vecinoActual.size() && !huboMejora;iter++){
				vecinoActual=solucion;
				if(vecinoActual[iter]+epsilon<=nUpperBound){
					vecinoActual[iter]+=epsilon;
					
					fitnessVecinoActual=selectFuncionObjetivo(nFunction,vecinoActual);
					contEvals++;
					contEvalsBL++;
					
					if(maximizacion){
						if(fitnessVecinoActual>fitnessSolucion){
							fitnessSolucion=fitnessVecinoActual;
							solucion=vecinoActual;
							huboMejora=true;
						}
					}else{
						if(fitnessVecinoActual<fitnessSolucion){
							fitnessSolucion=fitnessVecinoActual;
							solucion=vecinoActual;
							huboMejora=true;
						}	
					}
					
				}

				if(!huboMejora){
					vecinoActual=solucion;
					if(vecinoActual[iter]-epsilon>=nLowerBound){
						vecinoActual[iter]-=epsilon;
						
						fitnessVecinoActual=selectFuncionObjetivo(nFunction,vecinoActual);
						contEvals++;
						contEvalsBL++;
						
						if(maximizacion){
							if(fitnessVecinoActual>fitnessSolucion){
								fitnessSolucion=fitnessVecinoActual;
								solucion=vecinoActual;
								huboMejora=true;
							}
						}else{
							if(fitnessVecinoActual<fitnessSolucion){
								fitnessSolucion=fitnessVecinoActual;
								solucion=vecinoActual;
								huboMejora=true;
							}	
						}
					}
				}
			}
		}
	}	

	

	
	//FUNCION OBJETIVO: Michaelwicz
	/*double funcionObjetivo(const vector<double> &u) {
	    double result = 0.0;
	    for (int i = 0; i < u.size(); ++i) {
	        double a = sin(u[i]);
	        double b = sin(((i + 1) * u[i] * u[i]) / PI);
	        double c = pow(b, 20);
	        result += a * c;
	    }
	    return -1.0 * result;
	    //double z=-sin(u[0])*pow(sin( (u[0]*u[0])/PI ),20)-sin(u[1])*pow(sin( (2*u[1]*u[1])/PI ),20);
	    //return z;
	}*/


	double selectFuncionObjetivo(int nFunction,vector<double> &u){
		//return funciones.selectFuncionObjetivo(nFunction,u);
		cec14_test_func(&u[0],f,n,m,func_num);
		return f[0];
	}


	bool SolutionIsValid(vector<double> solucion){
		bool fallo=false;
		for(int i=0;i<solucion.size() && !fallo;i++){
			if(solucion[i]>nUpperBound || solucion[i]<nLowerBound){
				fallo=true;
			}
		}
		return fallo;
	}

public:

	

	CuckooSearch(int _numeroNidos, float _probabilidadDescubrimiento, /*float _tolerancia,*/ int _nd, int _lb, int _ub, int _maxEvals,
				float _lambda, float _epsilon, bool _maximizacion, float _probReinicializacion, int _nAgentes, int _maxEvalsBL){
		numeroNidos=_numeroNidos;	probabilidadDescubrimiento=_probabilidadDescubrimiento;
		//tolerancia=_tolerancia;		
		nd=_nd;
		//lowerBound=vector<int>(nd,_lb); //Inicializa el vector de cota inferior con nd valores de lo que indique _lb
		//upperBound=vector<int>(nd,_ub); //Inicializa el vector de cota superior con nd valores de lo que indique _ub
		nLowerBound=_lb; nUpperBound=_ub;
		maxEvals=_maxEvals;
		lambda=_lambda;
		epsilon=_epsilon;
		maximizacion=_maximizacion;
		probReinicializacion=_probReinicializacion;
		nAgentes=_nAgentes;
		maxEvalsBL=_maxEvalsBL;
	}




	void cuckooSearchAlgorithm(int nFunction,vector<double> &solucion, double& fitness){
		int contEvals=0;

		//vector<double> misFitness(maxEvals);
		//int punteroMisFitness=0;

		//Paso 1: Generación de soluciones aleatorias y me quedo con la mejor
		vector< vector<double> > poblacionInicial(numeroNidos);
		vector<double> fitnessPoblacionInicial(numeroNidos);

		for(int i=0;i<poblacionInicial.size();i++){
			poblacionInicial[i]	= generateRandomSolution();
			fitnessPoblacionInicial[i] = selectFuncionObjetivo(nFunction,poblacionInicial[i]);
			contEvals++;
		}

		

		vector<double> nuevoNido(nd);
		double fitnessNuevoNido;
		vector<double> nidoExistente(nd);
		double fitnessNidoExistente;
		
		vector<double> bestNido(nd);
		double fitnessBestNido;



		//Me quedo con la mejor solucion hasta ahora
		int indexBestNido=0;
		for(int i=1;i<poblacionInicial.size();i++){
			if(maximizacion){
				if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
					indexBestNido=i;
				}
			}else{
				if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
					indexBestNido=i;
				}
			}
			
			bestNido=poblacionInicial[indexBestNido];
			fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
		}

		//poblacionInicial[poblacionInicial.size()-1]=bestNido;
		//fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNido;	

		//misFitness[punteroMisFitness]=fitnessBestNido;
		//punteroMisFitness++;

		while (contEvals<maxEvals){
			//Paso 2: Diversidad con levy flights
			nuevoNido=levyFlight(poblacionInicial[(int)aleatorio(0,poblacionInicial.size()-0.00001)],bestNido);
			fitnessNuevoNido=selectFuncionObjetivo(nFunction,nuevoNido);
			contEvals++;
			int indexNidoExistente=(int)aleatorio(0,poblacionInicial.size()-0.00001);
			nidoExistente=poblacionInicial[indexNidoExistente];
			fitnessNidoExistente=fitnessPoblacionInicial[indexNidoExistente];
			
			if(maximizacion){
				if(fitnessNuevoNido>fitnessNidoExistente){ //Modificar aqui si se busca máximo o mínimo
					poblacionInicial[indexNidoExistente]=nuevoNido;
					fitnessPoblacionInicial[indexNidoExistente]=fitnessNuevoNido;
				}	
			}else{
				if(fitnessNuevoNido<fitnessNidoExistente){ //Modificar aqui si se busca máximo o mínimo
					poblacionInicial[indexNidoExistente]=nuevoNido;
					fitnessPoblacionInicial[indexNidoExistente]=fitnessNuevoNido;
				}	
			}
			


			

			//Paso 3: Descubrir y destruir los peores nidos, se crean nuevos.
			int descubiertos=probabilidadDescubrimiento*poblacionInicial.size();
			vector<bool> marcaPeores(poblacionInicial.size());
			int indexPeor=0;
			for(int vueltas=0;vueltas<descubiertos;vueltas++){
				for(int i=1;i<poblacionInicial.size();i++){
					if(maximizacion){
						if(fitnessPoblacionInicial[i]<fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
							indexPeor=i;
						}
					}else{
						if(fitnessPoblacionInicial[i]>fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
							indexPeor=i;
						}
					}
				}
				
				poblacionInicial[indexPeor]	= generateRandomSolution();
				fitnessPoblacionInicial[indexPeor] = selectFuncionObjetivo(nFunction,poblacionInicial[indexPeor]);
				contEvals++;

				marcaPeores[indexPeor]=true;
				indexPeor=0;
			}

			//Me quedo con la mejor solucion hasta ahora
			int indexBestNido=0;
			for(int i=1;i<poblacionInicial.size();i++){
				if(maximizacion){
					if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}
				}else{
					if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}	
				}
				
				bestNido=poblacionInicial[indexBestNido];
				fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
			}

			poblacionInicial[poblacionInicial.size()-1]=bestNido;
			fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNido;

			//misFitness[punteroMisFitness]=fitnessBestNido;
			//punteroMisFitness++;
		}		
		
		solucion=bestNido;
		fitness=fitnessBestNido;

		/*cout<<"GRAFICA DIVERSIDAD CONVERGENCIA"<<endl;
		for(int i=0;i<misFitness.size();i++){
			if(i%100==0 && misFitness[i]!=0)
				cout<<misFitness[i]<<",";
		}
		cout<<endl;*/

	}






















	//La V2 consiste en aplicar búsqueda local a los 10% mejores en cada generacion: TRIUNFO!!

	void cuckooSearchAlgorithmV2(int nFunction,vector<double> &solucion, double& fitness){
		int contEvals=0;
		//vector<double> misFitness(maxEvals);
		//int punteroMisFitness=0;

		//Paso 1: Generación de soluciones aleatorias y me quedo con la mejor
		vector< vector<double> > poblacionInicial(numeroNidos);
		vector<double> fitnessPoblacionInicial(numeroNidos);

		for(int i=0;i<poblacionInicial.size();i++){
			poblacionInicial[i]	= generateRandomSolution();
			fitnessPoblacionInicial[i] = selectFuncionObjetivo(nFunction,poblacionInicial[i]);
			contEvals++;
		}

		

		vector<double> nuevoNido(nd);
		double fitnessNuevoNido;
		vector<double> nidoExistente(nd);
		double fitnessNidoExistente;
		
		vector<double> bestNido(nd);
		double fitnessBestNido;



		//Me quedo con la mejor solucion hasta ahora
		int indexBestNido=0;
		for(int i=1;i<poblacionInicial.size();i++){
			if(maximizacion){
				if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
					indexBestNido=i;
				}
			}else{
				if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
					indexBestNido=i;
				}
			}
			
			bestNido=poblacionInicial[indexBestNido];
			fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
		}

		//poblacionInicial[poblacionInicial.size()-1]=bestNido;
		//fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNido;	

		//misFitness[punteroMisFitness]=fitnessBestNido;
		//punteroMisFitness++;


		while (contEvals<maxEvals){
			//Paso 2: Diversidad con levy flights
			nuevoNido=levyFlight(poblacionInicial[(int)aleatorio(0,poblacionInicial.size()-0.00001)],bestNido);
			fitnessNuevoNido=selectFuncionObjetivo(nFunction,nuevoNido);
			contEvals++;
			int indexNidoExistente=(int)aleatorio(0,poblacionInicial.size()-0.00001);
			nidoExistente=poblacionInicial[indexNidoExistente];
			fitnessNidoExistente=fitnessPoblacionInicial[indexNidoExistente];
			
			if(maximizacion){
				if(fitnessNuevoNido>fitnessNidoExistente){ //Modificar aqui si se busca máximo o mínimo
					poblacionInicial[indexNidoExistente]=nuevoNido;
					fitnessPoblacionInicial[indexNidoExistente]=fitnessNuevoNido;
				}	
			}else{
				if(fitnessNuevoNido<fitnessNidoExistente){ //Modificar aqui si se busca máximo o mínimo
					poblacionInicial[indexNidoExistente]=nuevoNido;
					fitnessPoblacionInicial[indexNidoExistente]=fitnessNuevoNido;
				}	
			}
			


			

			//Paso 3: Descubrir y destruir los peores nidos, se crean nuevos.
			int descubiertos=probabilidadDescubrimiento*poblacionInicial.size();
			vector<bool> marcaPeores(poblacionInicial.size());
			int indexPeor=0;
			for(int vueltas=0;vueltas<descubiertos;vueltas++){
				for(int i=1;i<poblacionInicial.size();i++){
					if(maximizacion){
						if(fitnessPoblacionInicial[i]<fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
							indexPeor=i;
						}
					}else{
						if(fitnessPoblacionInicial[i]>fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
							indexPeor=i;
						}
					}
				}
				
				poblacionInicial[indexPeor]	= generateRandomSolution();
				fitnessPoblacionInicial[indexPeor] = selectFuncionObjetivo(nFunction,poblacionInicial[indexPeor]);
				contEvals++;

				marcaPeores[indexPeor]=true;
				indexPeor=0;
			}

			//Me quedo con la mejor solucion hasta ahora
			int indexBestNido=0;
			
			
			for(int i=1;i<poblacionInicial.size();i++){
				if(maximizacion){
					if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}
				}else{
					if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}	
				}
				
				bestNido=poblacionInicial[indexBestNido];
				fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
			}

			poblacionInicial[poblacionInicial.size()-1]=bestNido;
			fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNido;



			//Aplicacion de busqueda local al 10% de mejores soluciones
			//int nIndBL=0.1*poblacionInicial.size();
			int nIndBL=1;
			vector<bool> marcaMejores(poblacionInicial.size());
			int indexMejor=0;
			for(int vueltas=0;vueltas<nIndBL;vueltas++){
				for(int i=1;i<poblacionInicial.size();i++){
					if(maximizacion){
						if(fitnessPoblacionInicial[i]>fitnessPoblacionInicial[indexMejor] && !marcaMejores[i]){ //Modificar aqui cuando una solucion es mejor que otra
							indexMejor=i;
						}
					}else{
						if(fitnessPoblacionInicial[i]<fitnessPoblacionInicial[indexMejor] && !marcaMejores[i]){ //Modificar aqui cuando una solucion es mejor que otra
							indexMejor=i;
						}
					}
				}
				localSearch(nFunction,poblacionInicial[indexMejor], fitnessPoblacionInicial[indexMejor], contEvals, maxEvalsBL);
				marcaMejores[indexMejor]=true;
				indexMejor=0;
			}
						
			//misFitness[punteroMisFitness]=fitnessBestNido;
			//punteroMisFitness++;		
		}		
		
		solucion=bestNido;
		fitness=fitnessBestNido;


		/*cout<<"GRAFICA DIVERSIDAD CONVERGENCIA"<<endl;
		for(int i=0;i<misFitness.size();i++){
			if(misFitness[i]!=0)
				cout<<misFitness[i]<<",";
		}
		cout<<endl;*/

	}






















	//La V3 mejora ciertos aspectos de elitismo, como guardar siempre la mejor solucion encontrada: FRACASO!!
	//Cuando se descubren los nidos quedarse con el mejor para no perderlo.
	void cuckooSearchAlgorithmV3(int nFunction,vector<double> &solucion, double& fitness){
		int contEvals=0;

		//vector<double> misFitness(maxEvals);
		//int punteroMisFitness=0;

		//Paso 1: Generación de soluciones aleatorias y me quedo con la mejor
		vector< vector<double> > poblacionInicial(numeroNidos);
		vector<double> fitnessPoblacionInicial(numeroNidos);

		for(int i=0;i<poblacionInicial.size();i++){
			poblacionInicial[i]	= generateRandomSolution();
			fitnessPoblacionInicial[i] = selectFuncionObjetivo(nFunction,poblacionInicial[i]);
			contEvals++;
		}

		

		vector<double> nuevoNido(nd);
		double fitnessNuevoNido;
		vector<double> nidoExistente(nd);
		double fitnessNidoExistente;
		
		vector<double> bestNido(nd);
		double fitnessBestNido;



		//Me quedo con la mejor solucion hasta ahora
		int indexBestNido=0;
		for(int i=1;i<poblacionInicial.size();i++){
			if(maximizacion){
				if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
					indexBestNido=i;
				}
			}else{
				if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
					indexBestNido=i;
				}
			}
			
			bestNido=poblacionInicial[indexBestNido];
			fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
		}

		//poblacionInicial[poblacionInicial.size()-1]=bestNido;
		//fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNido;	

		//misFitness[punteroMisFitness]=fitnessBestNido;
		//punteroMisFitness++;



		while (contEvals<maxEvals){
			//Paso 2: Diversidad con levy flights
			nuevoNido=levyFlight(poblacionInicial[(int)aleatorio(0,poblacionInicial.size()-0.00001)],bestNido);
			fitnessNuevoNido=selectFuncionObjetivo(nFunction,nuevoNido);
			contEvals++;
			//int indexNidoExistente=(int)aleatorio(0,poblacionInicial.size()-0.00001);
			//nidoExistente=poblacionInicial[indexNidoExistente];
			//fitnessNidoExistente=fitnessPoblacionInicial[indexNidoExistente];
			
			if(maximizacion){
				if(fitnessNuevoNido>fitnessBestNido){ //Modificar aqui si se busca máximo o mínimo
					poblacionInicial[indexBestNido]=nuevoNido;
					fitnessPoblacionInicial[indexBestNido]=fitnessNuevoNido;
					
					bestNido=poblacionInicial[indexBestNido];
					fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
				}	
			}else{
				if(fitnessNuevoNido<fitnessNidoExistente){ //Modificar aqui si se busca máximo o mínimo
					poblacionInicial[indexBestNido]=nuevoNido;
					fitnessPoblacionInicial[indexBestNido]=fitnessNuevoNido;
				
					bestNido=poblacionInicial[indexBestNido];
					fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
				}	
			}


			


			

			//Paso 3: Descubrir y destruir los peores nidos, se crean nuevos.
			int descubiertos=probabilidadDescubrimiento*poblacionInicial.size();
			vector<bool> marcaPeores(poblacionInicial.size());
			int indexPeor=0;
			for(int vueltas=0;vueltas<descubiertos;vueltas++){
				for(int i=1;i<poblacionInicial.size();i++){
					if(maximizacion){
						if(fitnessPoblacionInicial[i]<fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
							indexPeor=i;
						}
					}else{
						if(fitnessPoblacionInicial[i]>fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
							indexPeor=i;
						}
					}
				}
				
				poblacionInicial[indexPeor]	= generateRandomSolution();
				fitnessPoblacionInicial[indexPeor] = selectFuncionObjetivo(nFunction,poblacionInicial[indexPeor]);
				contEvals++;

				marcaPeores[indexPeor]=true;
				indexPeor=0;
			}

			//Meto el mejor para no perderlo
			poblacionInicial[poblacionInicial.size()-1]=bestNido;
			fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNido;

			//Me quedo con la mejor solucion hasta ahora, uno que haya entrado puede ser mejor
			indexBestNido=0;
			
			
			for(int i=1;i<poblacionInicial.size();i++){
				if(maximizacion){
					if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}
				}else{
					if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}	
				}
				
				bestNido=poblacionInicial[indexBestNido];
				fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
			}

			poblacionInicial[poblacionInicial.size()-1]=bestNido;
			fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNido;



			//Aplicacion de busqueda local al 10% de mejores soluciones
			//int nIndBL=0.1*poblacionInicial.size();
			int nIndBL=1;
			vector<bool> marcaMejores(poblacionInicial.size());
			int indexMejor=0;
			for(int vueltas=0;vueltas<nIndBL;vueltas++){
				for(int i=1;i<poblacionInicial.size();i++){
					if(maximizacion){
						if(fitnessPoblacionInicial[i]>fitnessPoblacionInicial[indexMejor] && !marcaMejores[i]){ //Modificar aqui cuando una solucion es mejor que otra
							indexMejor=i;
						}
					}else{
						if(fitnessPoblacionInicial[i]<fitnessPoblacionInicial[indexMejor] && !marcaMejores[i]){ //Modificar aqui cuando una solucion es mejor que otra
							indexMejor=i;
						}
					}
				}
				localSearch(nFunction,poblacionInicial[indexMejor], fitnessPoblacionInicial[indexMejor], contEvals, maxEvalsBL);
				marcaMejores[indexMejor]=true;
				indexMejor=0;
			}

			//misFitness[punteroMisFitness]=fitnessBestNido;
			//punteroMisFitness++;					
		}		
		
		solucion=bestNido;
		fitness=fitnessBestNido;

		/*cout<<"GRAFICA DIVERSIDAD CONVERGENCIA"<<endl;
		for(int i=0;i<misFitness.size();i++){
			if(misFitness[i]!=0)
				cout<<misFitness[i]<<",";
		}
		cout<<endl;*/

	}























	//La V4 parte de la V2 y se encarga de reinicializar la poblacion si no hay mejoras

	void cuckooSearchAlgorithmV4(int nFunction,vector<double> &solucion, double& fitness){
		int contEvals=0;

		//vector<double> misFitness(maxEvals);
		//int punteroMisFitness=0;

		//Paso 1: Generación de soluciones aleatorias y me quedo con la mejor
		vector< vector<double> > poblacionInicial(numeroNidos);
		vector<double> fitnessPoblacionInicial(numeroNidos);

		for(int i=0;i<poblacionInicial.size();i++){
			poblacionInicial[i]	= generateRandomSolution();
			fitnessPoblacionInicial[i] = selectFuncionObjetivo(nFunction,poblacionInicial[i]);
			contEvals++;
		}

		

		vector<double> nuevoNido(nd);
		double fitnessNuevoNido;
		vector<double> nidoExistente(nd);
		double fitnessNidoExistente;
		
		vector<double> bestNido(nd);
		double fitnessBestNido;


		vector<double> bestNidoGlobal(nd);
		double fitnessBestNidoGlobal;



		//Me quedo con la mejor solucion hasta ahora
		int indexBestNido=0;
		for(int i=1;i<poblacionInicial.size();i++){
			if(maximizacion){
				if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
					indexBestNido=i;
				}
			}else{
				if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
					indexBestNido=i;
				}
			}
			
			bestNido=poblacionInicial[indexBestNido];
			fitnessBestNido=fitnessPoblacionInicial[indexBestNido];

			bestNidoGlobal=poblacionInicial[indexBestNido];
			fitnessBestNidoGlobal=fitnessPoblacionInicial[indexBestNido];
		}

		//poblacionInicial[poblacionInicial.size()-1]=bestNidoGlobal;
		//fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNidoGlobal;	

		//misFitness[punteroMisFitness]=fitnessBestNido;
		//punteroMisFitness++;


		while (contEvals<maxEvals){
			//Paso 2: Diversidad con levy flights
			nuevoNido=levyFlight(poblacionInicial[(int)aleatorio(0,poblacionInicial.size()-0.00001)],bestNido);
			fitnessNuevoNido=selectFuncionObjetivo(nFunction,nuevoNido);
			contEvals++;
			//int indexNidoExistente=(int)aleatorio(0,poblacionInicial.size()-0.00001);
			//nidoExistente=poblacionInicial[indexNidoExistente];
			//fitnessNidoExistente=fitnessPoblacionInicial[indexNidoExistente];
			
			if(maximizacion){
				if(fitnessNuevoNido>fitnessNidoExistente){ //Modificar aqui si se busca máximo o mínimo
					poblacionInicial[indexBestNido]=nuevoNido;
					fitnessPoblacionInicial[indexBestNido]=fitnessNuevoNido;
				
					bestNido=poblacionInicial[indexBestNido];
					fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
				}

				if(fitnessBestNido>fitnessBestNidoGlobal){
					bestNidoGlobal=bestNido;
					fitnessBestNidoGlobal=fitnessBestNido;	
				}	
			}else{
				if(fitnessNuevoNido<fitnessNidoExistente){ //Modificar aqui si se busca máximo o mínimo
					poblacionInicial[indexBestNido]=nuevoNido;
					fitnessPoblacionInicial[indexBestNido]=fitnessNuevoNido;
				
					bestNido=poblacionInicial[indexBestNido];
					fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
				}

				if(fitnessBestNido<fitnessBestNidoGlobal){
					bestNidoGlobal=bestNido;
					fitnessBestNidoGlobal=fitnessBestNido;	
				}	
			}
			



			

			//Paso 3: Descubrir y destruir los peores nidos, se crean nuevos.
			int descubiertos=probabilidadDescubrimiento*poblacionInicial.size();
			vector<bool> marcaPeores(poblacionInicial.size());
			int indexPeor=0;
			for(int vueltas=0;vueltas<descubiertos;vueltas++){
				for(int i=1;i<poblacionInicial.size();i++){
					if(maximizacion){
						if(fitnessPoblacionInicial[i]<fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
							indexPeor=i;
						}
					}else{
						if(fitnessPoblacionInicial[i]>fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
							indexPeor=i;
						}
					}
				}
				
				poblacionInicial[indexPeor]	= generateRandomSolution();
				fitnessPoblacionInicial[indexPeor] = selectFuncionObjetivo(nFunction,poblacionInicial[indexPeor]);
				contEvals++;

				marcaPeores[indexPeor]=true;
				indexPeor=0;
			}

			//Me quedo con la mejor solucion hasta ahora
			int indexBestNido=0;
			
			
			for(int i=1;i<poblacionInicial.size();i++){
				if(maximizacion){
					if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}

					bestNido=poblacionInicial[indexBestNido];
					fitnessBestNido=fitnessPoblacionInicial[indexBestNido];

					if(fitnessBestNido>fitnessBestNidoGlobal){
						bestNidoGlobal=bestNido;
						fitnessBestNidoGlobal=fitnessBestNido;
					}
				}else{
					if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}

					bestNido=poblacionInicial[indexBestNido];
					fitnessBestNido=fitnessPoblacionInicial[indexBestNido];

					if(fitnessBestNido<fitnessBestNidoGlobal){
						bestNidoGlobal=bestNido;
						fitnessBestNidoGlobal=fitnessBestNido;
					}
				}
				

			}

			poblacionInicial[poblacionInicial.size()-1]=bestNidoGlobal;
			fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNidoGlobal;



			//Aplicacion de busqueda local al 10% de mejores soluciones
			//int nIndBL=0.1*poblacionInicial.size();
			int nIndBL=1;
			vector<bool> marcaMejores(poblacionInicial.size());
			int indexMejor=0;
			for(int vueltas=0;vueltas<nIndBL;vueltas++){
				for(int i=1;i<poblacionInicial.size();i++){
					if(maximizacion){
						if(fitnessPoblacionInicial[i]>fitnessPoblacionInicial[indexMejor] && !marcaMejores[i]){ //Modificar aqui cuando una solucion es mejor que otra
							indexMejor=i;
						}
					}else{
						if(fitnessPoblacionInicial[i]<fitnessPoblacionInicial[indexMejor] && !marcaMejores[i]){ //Modificar aqui cuando una solucion es mejor que otra
							indexMejor=i;
						}
					}
				}
				localSearch(nFunction,poblacionInicial[indexMejor], fitnessPoblacionInicial[indexMejor], contEvals, maxEvalsBL);
				marcaMejores[indexMejor]=true;
				indexMejor=0;
			}




				

			
			//Reinicializo la poblacion metiendo la mejor solucion hasta ahora
			if(probReinicializacion>=aleatorio(0,1)){
				//cout<<"Reinicializacion con mejor solucion -> "<<fitnessBestNidoGlobal<<endl;
				for(int i=0;i<poblacionInicial.size();i++){
					poblacionInicial[i]	= generateRandomSolution();
					fitnessPoblacionInicial[i] = selectFuncionObjetivo(nFunction,poblacionInicial[i]);
					contEvals++;
				}


				poblacionInicial[poblacionInicial.size()-1]=bestNidoGlobal;
				fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNidoGlobal;
			}

			//misFitness[punteroMisFitness]=fitnessBestNido;
			//punteroMisFitness++;
		}		
		
		solucion=bestNido;
		fitness=fitnessBestNido;

		/*cout<<"GRAFICA DIVERSIDAD CONVERGENCIA"<<endl;
		for(int i=0;i<misFitness.size();i++){
			if(misFitness[i]!=0)
				cout<<misFitness[i]<<",";
		}
		cout<<endl;*/

	}



















	//La V5 introduce la filosofía multiagente

	void cuckooSearchAlgorithmV5(int nFunction,vector<double> &solucion, double& fitness){
		vector<double> sd = generateRandomSolution();
		double fd = selectFuncionObjetivo(nFunction,sd);
		int contEvals=0, evalLS=0.3*maxEvals;
		//vector<double> misFitness(maxEvals);
		//int punteroMisFitness=0;

		for(int i=0;i<nAgentes;i++){
			//cout<<"Cuckoo "<<i<<" buscando..."<<endl;
			int contGeneraciones=1;

			//Paso 1: Generación de soluciones aleatorias y me quedo con la mejor
			vector< vector<double> > poblacionInicial(numeroNidos);
			vector<double> fitnessPoblacionInicial(numeroNidos);

			for(int i=0;i<poblacionInicial.size();i++){
				poblacionInicial[i]	= generateRandomSolution();
				fitnessPoblacionInicial[i] = selectFuncionObjetivo(nFunction,poblacionInicial[i]);
				contEvals++;
			}

			

			vector<double> nuevoNido(nd);
			double fitnessNuevoNido;
			vector<double> nidoExistente(nd);
			double fitnessNidoExistente;
			
			vector<double> bestNido(nd);
			double fitnessBestNido;


			vector<double> bestNidoGlobal(nd);
			double fitnessBestNidoGlobal;



			//Me quedo con la mejor solucion hasta ahora
			int indexBestNido=0;
			for(int i=1;i<poblacionInicial.size();i++){
				if(maximizacion){
					if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}
				}else{
					if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}
				}
				
				bestNido=poblacionInicial[indexBestNido];
				fitnessBestNido=fitnessPoblacionInicial[indexBestNido];

				bestNidoGlobal=poblacionInicial[indexBestNido];
				fitnessBestNidoGlobal=fitnessPoblacionInicial[indexBestNido];
			}

			//poblacionInicial[poblacionInicial.size()-1]=bestNidoGlobal;
			//fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNidoGlobal;	
			
			//misFitness[punteroMisFitness]=fitnessBestNido;
			//punteroMisFitness++;



			while (contEvals<((i+1)*(maxEvals)/nAgentes))
			{
				//Reinicializo la poblacion metiendo la mejor solucion hasta ahora
				if(probReinicializacion>=aleatorio(0,1)){
					//cout<<"Reinicializacion con mejor solucion -> "<<fitnessBestNidoGlobal<<endl;
					for(int i=0;i<poblacionInicial.size();i++){
						poblacionInicial[i]	= generateRandomSolution();
						fitnessPoblacionInicial[i] = selectFuncionObjetivo(nFunction,poblacionInicial[i]);
						contEvals++;
					}


					poblacionInicial[poblacionInicial.size()-1]=bestNidoGlobal;
					fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNidoGlobal;
				}




				//Paso 2: Diversidad con levy flights
				nuevoNido=levyFlight(poblacionInicial[(int)aleatorio(0,poblacionInicial.size()-0.00001)],bestNido);
				fitnessNuevoNido=selectFuncionObjetivo(nFunction,nuevoNido);
				contEvals++;
				//int indexNidoExistente=(int)aleatorio(0,poblacionInicial.size()-0.00001);
				//nidoExistente=poblacionInicial[indexNidoExistente];
				//fitnessNidoExistente=fitnessPoblacionInicial[indexNidoExistente];
				
				if(maximizacion){
					if(fitnessNuevoNido>fitnessNidoExistente){ //Modificar aqui si se busca máximo o mínimo
						poblacionInicial[indexBestNido]=nuevoNido;
						fitnessPoblacionInicial[indexBestNido]=fitnessNuevoNido;
					
						bestNido=poblacionInicial[indexBestNido];
						fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
					}

					if(fitnessBestNido>fitnessBestNidoGlobal){
						bestNidoGlobal=bestNido;
						fitnessBestNidoGlobal=fitnessBestNido;	
					}	
				}else{
					if(fitnessNuevoNido<fitnessNidoExistente){ //Modificar aqui si se busca máximo o mínimo
						poblacionInicial[indexBestNido]=nuevoNido;
						fitnessPoblacionInicial[indexBestNido]=fitnessNuevoNido;
					
						bestNido=poblacionInicial[indexBestNido];
						fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
					}

					if(fitnessBestNido<fitnessBestNidoGlobal){
						bestNidoGlobal=bestNido;
						fitnessBestNidoGlobal=fitnessBestNido;	
					}	
				}
				



				

				//Paso 3: Descubrir y destruir los peores nidos, se crean nuevos.
				int descubiertos=probabilidadDescubrimiento*poblacionInicial.size();
				vector<bool> marcaPeores(poblacionInicial.size());
				int indexPeor=0;
				for(int vueltas=0;vueltas<descubiertos;vueltas++){
					for(int i=1;i<poblacionInicial.size();i++){
						if(maximizacion){
							if(fitnessPoblacionInicial[i]<fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
								indexPeor=i;
							}
						}else{
							if(fitnessPoblacionInicial[i]>fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
								indexPeor=i;
							}
						}
					}
					
					poblacionInicial[indexPeor]	= generateRandomSolution();
					fitnessPoblacionInicial[indexPeor] = selectFuncionObjetivo(nFunction,poblacionInicial[indexPeor]);
					contEvals++;

					marcaPeores[indexPeor]=true;
					indexPeor=0;
				}

				//Me quedo con la mejor solucion hasta ahora
				int indexBestNido=0;
				
				
				for(int i=1;i<poblacionInicial.size();i++){
					if(maximizacion){
						if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
							indexBestNido=i;
						}

						bestNido=poblacionInicial[indexBestNido];
						fitnessBestNido=fitnessPoblacionInicial[indexBestNido];

						if(fitnessBestNido>fitnessBestNidoGlobal){
							bestNidoGlobal=bestNido;
							fitnessBestNidoGlobal=fitnessBestNido;
						}
					}else{
						if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
							indexBestNido=i;
						}

						bestNido=poblacionInicial[indexBestNido];
						fitnessBestNido=fitnessPoblacionInicial[indexBestNido];

						if(fitnessBestNido<fitnessBestNidoGlobal){
							bestNidoGlobal=bestNido;
							fitnessBestNidoGlobal=fitnessBestNido;
						}
					}
					

				}

				poblacionInicial[poblacionInicial.size()-1]=bestNidoGlobal;
				fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNidoGlobal;

				if(contEvals%evalLS == 0){
				//Aplicacion de busqueda local al 10% de mejores soluciones
				//int nIndBL=0.1*poblacionInicial.size();
				int nIndBL=1;
				vector<bool> marcaMejores(poblacionInicial.size());
				int indexMejor=0;
				for(int vueltas=0;vueltas<nIndBL;vueltas++){
					for(int i=1;i<poblacionInicial.size();i++){
						if(maximizacion){
							if(fitnessPoblacionInicial[i]>fitnessPoblacionInicial[indexMejor] && !marcaMejores[i]){ //Modificar aqui cuando una solucion es mejor que otra
								indexMejor=i;
							}
						}else{
							if(fitnessPoblacionInicial[i]<fitnessPoblacionInicial[indexMejor] && !marcaMejores[i]){ //Modificar aqui cuando una solucion es mejor que otra
								indexMejor=i;
							}
						}
					}
					localSearch(nFunction,poblacionInicial[indexMejor], fitnessPoblacionInicial[indexMejor], contEvals, maxEvalsBL);
					marcaMejores[indexMejor]=true;
					indexMejor=0;
				}
			
			

				}

				contGeneraciones++;
			}

			

			


			solucion=bestNido;
			fitness=fitnessBestNido;

			if(maximizacion){
				if(fitnessBestNido>fd){
					sd=bestNido;
					fd=fitnessBestNido;
				}
			}else{
				if(fitnessBestNido<fd){
					sd=bestNido;
					fd=fitnessBestNido;
				}
			}



			//misFitness[punteroMisFitness]=fitnessBestNido;
			//punteroMisFitness++;
		}


		solucion=sd;
		fitness=fd;


		/*cout<<"GRAFICA DIVERSIDAD CONVERGENCIA"<<endl;
		for(int i=0;i<misFitness.size();i++){
			if(misFitness[i]!=0)
				cout<<misFitness[i]<<",";
		}
		cout<<endl;*/

	}



// v6 introduce cruces y mutacion


void cuckooSearchAlgorithmV6(int nFunction,vector<double> &solucion, double& fitness){
	vector<double> sd = generateRandomSolution();
	double fd = selectFuncionObjetivo(nFunction,sd);
	int contEvals=0, evalLS=0.5*maxEvals;
	//vector<double> misFitness(maxEvals);
	//int punteroMisFitness=0;

	for(int i=0;i<nAgentes;i++){
		//cout<<"Cuckoo "<<i<<" buscando..."<<endl;
		int contGeneraciones=1;

		//Paso 1: Generación de soluciones aleatorias y me quedo con la mejor
		vector< vector<double> > poblacionInicial(numeroNidos);
		vector<double> fitnessPoblacionInicial(numeroNidos);

		for(int i=0;i<poblacionInicial.size();i++){
			poblacionInicial[i]	= generateRandomSolution();
			fitnessPoblacionInicial[i] = selectFuncionObjetivo(nFunction,poblacionInicial[i]);
			contEvals++;
		}

		

		vector<double> nuevoNido(nd);
		double fitnessNuevoNido;
		vector<double> nidoExistente(nd);
		double fitnessNidoExistente;
		
		vector<double> bestNido(nd);
		double fitnessBestNido;


		vector<double> bestNidoGlobal(nd);
		double fitnessBestNidoGlobal;



		//Me quedo con la mejor solucion hasta ahora
		int indexBestNido=0;
		for(int i=1;i<poblacionInicial.size();i++){
			if(maximizacion){
				if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
					indexBestNido=i;
				}
			}else{
				if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
					indexBestNido=i;
				}
			}
			
			bestNido=poblacionInicial[indexBestNido];
			fitnessBestNido=fitnessPoblacionInicial[indexBestNido];

			bestNidoGlobal=poblacionInicial[indexBestNido];
			fitnessBestNidoGlobal=fitnessPoblacionInicial[indexBestNido];
		}

		//poblacionInicial[poblacionInicial.size()-1]=bestNidoGlobal;
		//fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNidoGlobal;	
		
		//misFitness[punteroMisFitness]=fitnessBestNido;
		//punteroMisFitness++;



		while (contEvals<((i+1)*(maxEvals)/nAgentes))
		{
			//Reinicializo la poblacion metiendo la mejor solucion hasta ahora
			if(probReinicializacion>=aleatorio(0,1)){
				//cout<<"Reinicializacion con mejor solucion -> "<<fitnessBestNidoGlobal<<endl;
				for(int i=0;i<poblacionInicial.size();i++){
					poblacionInicial[i]	= generateRandomSolution();
					fitnessPoblacionInicial[i] = selectFuncionObjetivo(nFunction,poblacionInicial[i]);
					contEvals++;
				}


				poblacionInicial[poblacionInicial.size()-1]=bestNidoGlobal;
				fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNidoGlobal;
			}




			//Paso 2: Diversidad con levy flights
			nuevoNido=levyFlight(poblacionInicial[(int)aleatorio(0,poblacionInicial.size()-0.00001)],bestNido);
			fitnessNuevoNido=selectFuncionObjetivo(nFunction,nuevoNido);
			contEvals++;
			//int indexNidoExistente=(int)aleatorio(0,poblacionInicial.size()-0.00001);
			//nidoExistente=poblacionInicial[indexNidoExistente];
			//fitnessNidoExistente=fitnessPoblacionInicial[indexNidoExistente];
			
			if(maximizacion){
				if(fitnessNuevoNido>fitnessNidoExistente){ //Modificar aqui si se busca máximo o mínimo
					poblacionInicial[indexBestNido]=nuevoNido;
					fitnessPoblacionInicial[indexBestNido]=fitnessNuevoNido;
				
					bestNido=poblacionInicial[indexBestNido];
					fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
				}

				if(fitnessBestNido>fitnessBestNidoGlobal){
					bestNidoGlobal=bestNido;
					fitnessBestNidoGlobal=fitnessBestNido;	
				}	
			}else{
				if(fitnessNuevoNido<fitnessNidoExistente){ //Modificar aqui si se busca máximo o mínimo
					poblacionInicial[indexBestNido]=nuevoNido;
					fitnessPoblacionInicial[indexBestNido]=fitnessNuevoNido;
				
					bestNido=poblacionInicial[indexBestNido];
					fitnessBestNido=fitnessPoblacionInicial[indexBestNido];
				}

				if(fitnessBestNido<fitnessBestNidoGlobal){
					bestNidoGlobal=bestNido;
					fitnessBestNidoGlobal=fitnessBestNido;	
				}	
			}
			



			

			//Paso 3: Descubrir y destruir los peores nidos, se crean nuevos.
			int descubiertos=probabilidadDescubrimiento*poblacionInicial.size();
			vector<bool> marcaPeores(poblacionInicial.size());
			int indexPeor=0;
			for(int vueltas=0;vueltas<descubiertos;vueltas++){
				for(int i=1;i<poblacionInicial.size();i++){
					if(maximizacion){
						if(fitnessPoblacionInicial[i]<fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
							indexPeor=i;
						}
					}else{
						if(fitnessPoblacionInicial[i]>fitnessPoblacionInicial[indexPeor] && !marcaPeores[i]){ //Modificar aqui cuando una solucion es peor que otra
							indexPeor=i;
						}
					}
				}
				
				poblacionInicial[indexPeor]	= generateRandomSolution();
				fitnessPoblacionInicial[indexPeor] = selectFuncionObjetivo(nFunction,poblacionInicial[indexPeor]);
				contEvals++;

				marcaPeores[indexPeor]=true;
				indexPeor=0;
			}

			//Me quedo con la mejor solucion hasta ahora
			int indexBestNido=0;
			
			
			for(int i=1;i<poblacionInicial.size();i++){
				if(maximizacion){
					if(fitnessPoblacionInicial[indexBestNido]<fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}

					bestNido=poblacionInicial[indexBestNido];
					fitnessBestNido=fitnessPoblacionInicial[indexBestNido];

					if(fitnessBestNido>fitnessBestNidoGlobal){
						bestNidoGlobal=bestNido;
						fitnessBestNidoGlobal=fitnessBestNido;
					}
				}else{
					if(fitnessPoblacionInicial[indexBestNido]>fitnessPoblacionInicial[i]){
						indexBestNido=i;
					}

					bestNido=poblacionInicial[indexBestNido];
					fitnessBestNido=fitnessPoblacionInicial[indexBestNido];

					if(fitnessBestNido<fitnessBestNidoGlobal){
						bestNidoGlobal=bestNido;
						fitnessBestNidoGlobal=fitnessBestNido;
					}
				}
				

			}

			poblacionInicial[poblacionInicial.size()-1]=bestNidoGlobal;
			fitnessPoblacionInicial[poblacionInicial.size()-1]=fitnessBestNidoGlobal;


       



       	//cout<<"antes: "<<contGeneraciones<<endl;
		contGeneraciones++;
		//cout<<contGeneraciones<<endl;

		//cruce y mutacion
		/////////////////////////////////

		if(contGeneraciones==50)
		{
			//cout<<"hola"<<endl;
			int tam_pob=poblacionInicial.size();
			int n_cruzan=tam_pob;
//			int mutan=(rand()/RAND_MAX)*tam_pob;
//			int mutan=5;

			//cruzan todos
			for(int i=0;i<n_cruzan;i++)
			{
				//obtenemos los puntos de cruce
				int r1,r2;
				//de esta forma obtenmos que r2 > r1
				r1=(rand()/RAND_MAX)*nd;
				r2=(rand()/RAND_MAX)*(nd-r1);
				r2+=r1;

				//r1=(rand()/RAND_MAX)*(nd/2); //obtenemos un aleatorio entre 0 y tam/2 de sol
				//r2=(rand()/RAND_MAX)*(nd/2); //obtenemos un aleatorio entre 0 y tam/2 de sol
				//r1=(rand()/RAND_MAX)*nd; //obtenemos un aleatorio entre 0 y tam de sol
				//r2=(r1+((rand()/RAND_MAX)*nd))%nd; //obtenemos un aleatorio entre 0 y tam de sol
				
				vector<double> copia=poblacionInicial[i];
				vector<double> copia2=poblacionInicial[(i+1)%tam_pob];
				for (int j = r1; j < r2; ++j)
				{
					copia[j]=poblacionInicial[(i+1)%tam_pob][j];
				    copia2[j]=poblacionInicial[i][j];
				}
				poblacionInicial[i]=copia;
				poblacionInicial[(i+1)%tam_pob]=copia2;

				//mutacion
				// int mutar=(rand()/RAND_MAX);
				// if(mutar>=0.001)
				// {
				// 	int aleatorio_mutacion_uno = (rand()/RAND_MAX)*tam_pob;
				// 	int aleatorio_mutacion_dos = (rand()/RAND_MAX)*nd;
				// 	int inicio=-100,fin=100;
				// 	poblacionInicial[aleatorio_mutacion_uno][aleatorio_mutacion_dos]=inicio+((rand()/RAND_MAX)*fin);
				// }

				//Evaluamos
				fitnessPoblacionInicial[i]=selectFuncionObjetivo(nFunction,poblacionInicial[i]);
				fitnessPoblacionInicial[(i+1)%tam_pob]=selectFuncionObjetivo(nFunction,poblacionInicial[(i+1)%tam_pob]);
			}

			/*
			//cruzan n aleatorios
			for (int i = 0; i < 10; ++i)
			{
				//obtenemos los que cruzan
				int cruza_uno,cruza_dos;
				cruza_uno=(rand()/RAND_MAX)*tam_pob;
				cruza_dos=(rand()/RAND_MAX)*(tam_pob-cruza_uno);
				cruza_dos+=cruza_uno;
				//obtenemos los puntos de cruce
				int r1,r2;
				//de esta forma obtenmos que r2 > r1
				r1=(rand()/RAND_MAX)*nd;
				r2=(rand()/RAND_MAX)*(nd-r1);
				r2+=r1;

				//r1=(rand()/RAND_MAX)*(nd/2); //obtenemos un aleatorio entre 0 y tam/2 de sol
				//r2=(rand()/RAND_MAX)*(nd/2); //obtenemos un aleatorio entre 0 y tam/2 de sol
				//r1=(rand()/RAND_MAX)*nd; //obtenemos un aleatorio entre 0 y tam de sol
				//r2=(r1+((rand()/RAND_MAX)*nd))%nd; //obtenemos un aleatorio entre 0 y tam de sol
				
				vector<double> copia=poblacionInicial[i];
				vector<double> copia2=poblacionInicial[(i+1)%tam_pob];
				for (int j = r1; j < r2; ++j)
				{
					copia[j]=poblacionInicial[cruza_dos][j];
				    copia2[j]=poblacionInicial[cruza_uno][j];
				}
				poblacionInicial[cruza_uno]=copia;
				poblacionInicial[cruza_dos]=copia2;
				
			}
			

*/


			/*
			//cruzan 3 peores
			for (int i = 0; i < mutan; ++i)
			{
				//obtenemos los que cruzan
				int cruza_uno,cruza_dos;
				cruza_uno=(rand()/RAND_MAX)*tam_pob;
				cruza_dos=(rand()/RAND_MAX)*(tam_pob-cruza_uno);
				cruza_dos+=cruza_uno;
				//obtenemos los puntos de cruce
				int r1,r2;
				//de esta forma obtenmos que r2 > r1
				r1=(rand()/RAND_MAX)*nd;
				r2=(rand()/RAND_MAX)*(nd-r1);
				r2+=r1;

				//r1=(rand()/RAND_MAX)*(nd/2); //obtenemos un aleatorio entre 0 y tam/2 de sol
				//r2=(rand()/RAND_MAX)*(nd/2); //obtenemos un aleatorio entre 0 y tam/2 de sol
				//r1=(rand()/RAND_MAX)*nd; //obtenemos un aleatorio entre 0 y tam de sol
				//r2=(r1+((rand()/RAND_MAX)*nd))%nd; //obtenemos un aleatorio entre 0 y tam de sol
				
				vector<double> copia=poblacionInicial[i];
				vector<double> copia2=poblacionInicial[(i+1)%tam_pob];
				for (int j = r1; j < r2; ++j)
				{
					copia[j]=poblacionInicial[cruza_dos][j];
				    copia2[j]=poblacionInicial[cruza_uno][j];
				}
				poblacionInicial[cruza_uno]=copia;
				poblacionInicial[cruza_dos]=copia2;
				
			}

			*/

			contGeneraciones=0;
		}

		/////////////////////////////////

		if(contEvals%evalLS == 0)
		{
			//Aplicacion de busqueda local al 10% de mejores soluciones
			//int nIndBL=0.1*poblacionInicial.size();
			int nIndBL=1;
			vector<bool> marcaMejores(poblacionInicial.size());
			int indexMejor=0;
			for(int vueltas=0;vueltas<nIndBL;vueltas++){
				for(int i=1;i<poblacionInicial.size();i++){
					if(maximizacion){
						if(fitnessPoblacionInicial[i]>fitnessPoblacionInicial[indexMejor] && !marcaMejores[i]){ //Modificar aqui cuando una solucion es mejor que otra
							indexMejor=i;
						}
					}else{
						if(fitnessPoblacionInicial[i]<fitnessPoblacionInicial[indexMejor] && !marcaMejores[i]){ //Modificar aqui cuando una solucion es mejor que otra
							indexMejor=i;
						}
					}
				}
				localSearch(nFunction,poblacionInicial[indexMejor], fitnessPoblacionInicial[indexMejor], contEvals, maxEvalsBL);
				marcaMejores[indexMejor]=true;
				indexMejor=0;
			}
			
		}



		}






		solucion=bestNido;
		fitness=fitnessBestNido;

		if(maximizacion){
			if(fitnessBestNido>fd){
				sd=bestNido;
				fd=fitnessBestNido;
			}
		}else{
			if(fitnessBestNido<fd){
				sd=bestNido;
				fd=fitnessBestNido;
			}
		}



		//misFitness[punteroMisFitness]=fitnessBestNido;
		//punteroMisFitness++;
	}


	solucion=sd;
	fitness=fd;


	/*cout<<"GRAFICA DIVERSIDAD CONVERGENCIA"<<endl;
	for(int i=0;i<misFitness.size();i++){
		if(misFitness[i]!=0)
			cout<<misFitness[i]<<",";
	}
	cout<<endl;*/

}














	double evaluarResultados(int nFunction, vector<double> &solucion){
		selectFuncionObjetivo(nFunction,solucion);
	}
};














int main(){

	/*int funcionAOptimizar=11;
	int problem_dimension=2;
	
	cout<<"Introduce funcion:";
	cin>>funcionAOptimizar;

	cout<<"Introduce dimension:";
	cin>>problem_dimension;
	

	srand (199);
	//cuckooSearch(int _numeroNidos, float _probabilidadDescubrimiento,
	//int dimensionProblema, int _lb, int _ub, int max_eval, float lambda, float epsilon, bool maximizacion, 
	//float probReinicializacion, float nAgentes, int maxEvalsBL)
	CuckooSearch cs(25, 0.25, problem_dimension, -100, 100,150000,2,0.001,false,0.1,10,0);
	
	vector<double> means(problem_dimension);
	double meanFitness=0;
	int iterations=20;
	vector<double> solucion;
	double fitness;
	


	// main del test
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////


    FILE *fpt;
    char FileName[30];
    m=2;
    n=problem_dimension;
    x=(double *)malloc(m*n*sizeof(double));
    f=(double *)malloc(sizeof(double)  *  m);
    // for (i = 0; i < 30; i++)
    // {
        func_num=funcionAOptimizar;
        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
        fpt = fopen(FileName,"r");
        if (fpt==NULL)
        {
            printf("\n Error: Cannot open input file for reading \n");
        }
       
        if (x==NULL)
            printf("\nError: there is insufficient memory available!\n");

        for(k=0;k<n;k++)
        {
                //fscanf(fpt,"%Lf",(long double*)&x[k]);
                float r;
                fscanf(fpt,"%f",&r);
                x[k]=r;
                //cout<<"k:"<<x[k]<<" "<<r<<endl;
                //printf("%Lf\n",x[k]);
        }

        fclose(fpt);

            for (j = 0; j < n; j++)
            {
                x[1*n+j]=0.0;
                //printf("%Lf\n",x[1*n+j]);
            }

    // }
   
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////







	



	//vector<double> resultadosEjecuciones(iterations);



	for(int i=0;i<iterations;i++){		
		cs.cuckooSearchAlgorithm(funcionAOptimizar,solucion,fitness);
		
		//cout<<"(";
		for(int i=0;i<solucion.size();i++){
			means[i]+=solucion[i];
			//cout<<solucion[i]<<" ";
		}
		//cout<<") -> "<<fitness<<endl;
		meanFitness+=fitness;
		//resultadosEjecuciones[i]=fitness;
	}
	
	cout<<"La solucion, en media, es: (";
	for(int i=0;i<solucion.size();i++){
		means[i]/=iterations;
		cout<<means[i]<<" ";
	}
	cout<<") -> "<<meanFitness/iterations<<endl;
	//cout<<") -> "<<cs.evaluarResultados(funcionAOptimizar,means)<<endl;



	//SALIDA FORMATO R
	//cout<<endl<<endl;
	//for(int i=0;i<iterations;i++){		
	//	cout<<resultadosEjecuciones[i]<<",";
	//}
	//cout<<endl<<endl;
	*/
	srand (199);
	int nNidos=25;
	int probDescubrimiento=0.25;
	int minRange=-100;
	int maxRange=100;
	int nEvaluaciones=100000;
	float lambda=2;
	float epsilon=0.001;
	bool maximizacion=false;
	float probReinicializacion=0.1;
	int nAgentes=10;
	int maxEvalsBL=0.10*nEvaluaciones;

	
	int funcionAOptimizar=1;
	int problem_dimension=10;

	/*
	cout<<"---------------VERSION ORIGINAL DEL ALGORITMO CUCKOO SEARCH---------------"<<endl;
	nEvaluaciones=100000;
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=20;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);
	        func_num=funcionAOptimizar;
	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithm(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}






	
	funcionAOptimizar=1;
	problem_dimension=30;
	nEvaluaciones=300000;
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=20;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);
	        func_num=funcionAOptimizar;
	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithm(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}*/














/*
	cout<<"---------------VERSION 2 DEL ALGORITMO CUCKOO SEARCH---------------"<<endl;

	
	funcionAOptimizar=1;
	problem_dimension=10;
	nEvaluaciones=100000;
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=20;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);
	        func_num=funcionAOptimizar;
	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithmV2(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}






	funcionAOptimizar=1;
	problem_dimension=30;
	nEvaluaciones=300000;
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=20;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);
	        func_num=funcionAOptimizar;
	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithmV2(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}













	cout<<"---------------VERSION 3 DEL ALGORITMO CUCKOO SEARCH---------------"<<endl;

	
	funcionAOptimizar=1;
	problem_dimension=10;
	nEvaluaciones=100000;
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=20;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);
	        func_num=funcionAOptimizar;
	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithmV3(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}





	funcionAOptimizar=1;
	problem_dimension=30;
	nEvaluaciones=300000;
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=20;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);
	        func_num=funcionAOptimizar;
	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithmV3(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}

*/

	














/*
	cout<<"---------------VERSION 4 DEL ALGORITMO CUCKOO SEARCH---------------"<<endl;

	
	funcionAOptimizar=1;
	problem_dimension=10;	
	nEvaluaciones=100000;
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=20;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);
	        func_num=funcionAOptimizar;
	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithmV4(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}






	funcionAOptimizar=1;
	problem_dimension=30;
	nEvaluaciones=300000;	
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=20;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);
	        func_num=funcionAOptimizar;
	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithmV4(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}




















*/

/*
	cout<<"---------------VERSION 5 DEL ALGORITMO CUCKOO SEARCH---------------"<<endl;

	
	funcionAOptimizar=1;
	problem_dimension=10;
	nEvaluaciones=100000;
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=25;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);
	        func_num=17;
	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithmV5(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}


*/

/*

	funcionAOptimizar=1;
	problem_dimension=30;
	nEvaluaciones=300000;	
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=20;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);
	        func_num=funcionAOptimizar;
	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithmV5(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}


*/


		cout<<"---------------VERSION 6 DEL ALGORITMO CUCKOO SEARCH---------------"<<endl;

	
	funcionAOptimizar=1;
	problem_dimension=10;
	nEvaluaciones=100000;
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=25;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);




	        func_num=funcionAOptimizar;







	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithmV6(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}






	funcionAOptimizar=1;
	problem_dimension=30;
	nEvaluaciones=300000;	
	while(funcionAOptimizar<=30){

		//cout<<"Funcion: "<< funcionAOptimizar << ". Dimension: "<< problem_dimension<<" -> ";

		CuckooSearch cs(nNidos, probDescubrimiento, problem_dimension, minRange, maxRange,nEvaluaciones,lambda,epsilon,maximizacion,probReinicializacion,nAgentes,maxEvalsBL);
		vector<double> means(problem_dimension);
		double meanFitness=0;
		int iterations=20;
		vector<double> solucion;
		double fitness;
		

	    FILE *fpt;
	    char FileName[30];
	    m=2;
	    n=problem_dimension;
	    x=(double *)malloc(m*n*sizeof(double));
	    f=(double *)malloc(sizeof(double)  *  m);
	        func_num=funcionAOptimizar;
	        sprintf(FileName, "./input_data/shift_data_%d.txt", func_num);
	        fpt = fopen(FileName,"r");
	        if (fpt==NULL)
	        {
	            printf("\n Error: Cannot open input file for reading \n");
	        }
	       
	        if (x==NULL)
	            printf("\nError: there is insufficient memory available!\n");

	        for(k=0;k<n;k++)
	        {
	                float r;
	                fscanf(fpt,"%f",&r);
	                x[k]=r;
	        }

	        fclose(fpt);

	            for (j = 0; j < n; j++)
	            {
	                x[1*n+j]=0.0;
	            }


		for(int i=0;i<iterations;i++){		
			cs.cuckooSearchAlgorithmV6(funcionAOptimizar,solucion,fitness);
			for(int i=0;i<solucion.size();i++){
				means[i]+=solucion[i];
			}
			meanFitness+=fitness;
		}
		
		//cout<<"La solucion, en media, es: (";
		//for(int i=0;i<solucion.size();i++){
		//	means[i]/=iterations;
		//	cout<<means[i]<<" ";
		//}
		//cout<<") -> "<<meanFitness/iterations<<endl;
		cout<<meanFitness/iterations<<endl;
		funcionAOptimizar++;
	}




	
}