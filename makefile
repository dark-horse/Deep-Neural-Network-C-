COMPILER = g++.exe
COMPILER_FLAGS = -O2 -g -Wall #-g for debug -O2 for optimise -Wall for additional messages
#COMPILER_FLAGS = -g -Wall -fpermissive #-g for debug -O2 for optimise -Wall for additional messages -fpermissive just to use labels in the main function
LINKER = ld.exe

main.exe : main.c deep_neural_network.cpp deep_neural_network.h
	${COMPILER} ${COMPILER_FLAGS} main.c deep_neural_network.cpp -o main.exe
	
main_test.exe : main_test.c deep_neural_network.cpp deep_neural_network.h
	${COMPILER} ${COMPILER_FLAGS} main_test.c deep_neural_network.cpp -o main_test.exe
clean :
	rm -rf *.o
	rm -rf *.exe
	rm -rf *.out
	rm -rf *.out1