# Makefile for spherIC

# Executable

BASE    = spherIC
EXT     = 
EXE     = spheric

# Compiler stuff

CC     = h5cc
CFLAGS = -O2 -Wall -I$(TACC_GSL_INC) -L$(TACC_GSL_LIB) -I$(TACC_HDF5_INC) -L$(TACC_HDF5_LIB)
LIBS     =
STAMPEDE3 = $(strip $(shell uname -a | grep -o stampede3))
ifneq ($(STAMPEDE3),)
# note bridges2 currently requires the following modules to be loaded:
# gcc/10.2.0  
# openmpi/4.0.5-gcc10.2.0
# phdf5/1.10.7-openmpi4.0.5-gcc10.2.0
# Other combinations may work but are not guaranteed
# CC      = mpicc
LIBS    = -L$(LIBRARY_PATH) -I$(INCLUDE)
LIBS    += -lhdf5 -lhdf5_hl -lz -ldl -lm
endif

LIBS    += -lm -lgsl -lgslcblas 

# Object definition

OBJ	= $(EXE).o functions.o routines.o io.o

# Rules

$(EXE):	$(OBJ) Makefile
	$(CC) $(CFLAGS) $(OBJ) -o $(EXE) $(LIBS)

.PHONY: clean
clean:
	-rm -f *.o *~ $(EXE)

tar:
	cd ..; tar cvf - $(BASE)$(EXT)/*.c $(BASE)$(EXT)/*.h $(BASE)$(EXT)/Makefile $(BASE)$(EXT)/doc  > $(BASE)$(EXT).tar

# Dependencies

spheric.o: definitions.h functions.h routines.h io.h
functions.o: definitions.h functions.h routines.h
routines.o: definitions.h functions.h routines.h
io.o: io.h
