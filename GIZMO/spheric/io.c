/* 
** io.c
**
** IO routines for spherIC_0.1
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "definitions.h"
#include "io.h"
#include <hdf5.h>
#include <hdf5_hl.h>


/*
** Routine for writing grid in r to an ASCII file.
*/

void write_gridr(FILE *file,const SI *si) {

  INT i,halo_flag,stars_flag;
  GRIDR *gridr;
	
  gridr = si->gridr;
  halo_flag = si->halo_flag;
  stars_flag = si->stars_flag;
	
  if(halo_flag == 1 && stars_flag == 1){
    fprintf(file,"#     0             1             2             3             4             5             6             7\n");
    fprintf(file,"#     r          rhoHalo       rhoStar         rho         MencHalo      MencStar        Menc          Pot\n");
    for (i = 0; i < NGRIDR; i++) {
      fprintf(file,OFD2" "OFD2" "OFD2" "OFD2" "OFD2" "OFD2" "OFD2" "OFD2"\n",
	      gridr->r[i],gridr->rhoHalo[i],gridr->rhoStar[i],gridr->rho[i],gridr->MencHalo[i],gridr->MencStar[i],gridr->Menc[i],gridr->Pot[i]);
    }
  }
	
  if(halo_flag == 1 && stars_flag == 0){
    fprintf(file,"#     0             1             2             3\n");
    fprintf(file,"#     r          rhoHalo       MencHalo        Pot\n");
    for (i = 0; i < NGRIDR; i++) {
      fprintf(file,OFD2" "OFD2" "OFD2" "OFD2"\n",
	      gridr->r[i],gridr->rhoHalo[i],gridr->MencHalo[i],gridr->Pot[i]);
    }
  }
	
  if(halo_flag == 0 && stars_flag == 1){
    fprintf(file,"#     0             1             2             3\n");
    fprintf(file,"#     r          rhoStar       MencStar        Pot\n");
    for (i = 0; i < NGRIDR; i++) {
      fprintf(file,OFD2" "OFD2" "OFD2" "OFD2"\n",
	      gridr->r[i],gridr->rhoStar[i],gridr->MencStar[i],gridr->Pot[i]);
    }
  }
	
}	

/*
** Routine for writing grid in f (distribution function) to an ASCII file.
*/

void write_griddf(FILE *file, const SI *si) {

  INT i,halo_flag,stars_flag;
	
  halo_flag = si->halo_flag;
  stars_flag = si->stars_flag;
  GRIDDF *griddf;
	
	
  if(halo_flag == 1 && stars_flag == 1){
    fprintf(file,"#     0             1             2             3             4\n");
    fprintf(file,"#     r             E             f           fstar          gE\n");
    griddf = si->griddf;
    for (i = 0; i < NGRIDDF; i++) {
      fprintf(file,OFD2" "OFD2" "OFD2" "OFD2" "OFD2"\n",
	      griddf->r[i],griddf->E[i],griddf->fE[i],griddf->fEstar[i],griddf->gE[i]);
    }
  }

  if(halo_flag == 1 && stars_flag == 0){
    fprintf(file,"#     0             1             2             3\n");
    fprintf(file,"#     r             E           fhalo          gE\n");
    griddf = si->griddf;
    for (i = 0; i < NGRIDDF; i++) {
      fprintf(file,OFD2" "OFD2" "OFD2" "OFD2"\n",
	      griddf->r[i],griddf->E[i],griddf->fE[i],griddf->gE[i]);
    }
  }
	
  if(halo_flag == 0 && stars_flag == 1){
    fprintf(file,"#     0             1             2             3\n");
    fprintf(file,"#     r             E           fstar          gE\n");
    griddf = si->griddf;
    for (i = 0; i < NGRIDDF; i++) {
      fprintf(file,OFD2" "OFD2" "OFD2" "OFD2"\n",
	      griddf->r[i],griddf->E[i],griddf->fEstar[i],griddf->gE[i]);
    }
  }
}


/*
** Routine for writing ifrit binary with positions and star_flag.
*/

void write_ifrit(FILE *F,const SI *si){

  int dummy,n,i,Nstar,N;
  float xl, xh,temp; 
  xh = si->sp->rs*10;
  xl = -xh;
   
  N = si->N;
  Nstar = si->Nstar;
   
  if(si->nostarpot_flag == 1 || si->stars_flag == 0){   
    n = N;
    Nstar = 0;
  } else {
    n = Nstar + N;
  }
   
  dummy = sizeof(int);
  assert(fwrite(&dummy,sizeof(int),1,F) == 1);
  assert(fwrite(&n,sizeof(int),1,F) == 1);
  assert(fwrite(&dummy,sizeof(int),1,F) == 1);
   
  dummy = 6*sizeof(float);
  assert(fwrite(&dummy,sizeof(int),1,F) == 1);
  assert(fwrite(&xl,sizeof(float),1,F) == 1);
  assert(fwrite(&xl,sizeof(float),1,F) == 1);
  assert(fwrite(&xl,sizeof(float),1,F) == 1);
  assert(fwrite(&xh,sizeof(float),1,F) == 1);
  assert(fwrite(&xh,sizeof(float),1,F) == 1);
  assert(fwrite(&xh,sizeof(float),1,F) == 1);
  assert(fwrite(&dummy,sizeof(int),1,F) == 1);
   
  dummy = n*sizeof(float);
  assert(fwrite(&dummy,sizeof(int),1,F) == 1) ;
  for(i = 0; i < N; i++) {
    temp = si->p[i].r[1];
    assert(fwrite(&temp,sizeof(float),1,F) == 1);
  }   
  for(i = 0; i < Nstar; i++) {
    temp = si->pstar[i].r[1];
    assert(fwrite(&temp,sizeof(float),1,F) == 1);
  }
  assert(fwrite(&dummy,sizeof(int),1,F) == 1);
   
  assert(fwrite(&dummy,sizeof(int),1,F) == 1) ;
  for(i = 0; i < N; i++) {
    temp = si->p[i].r[2];
    assert(fwrite(&temp,sizeof(float),1,F) == 1);
  }
  for(i = 0; i < Nstar; i++) {
    temp = si->pstar[i].r[2];
    assert(fwrite(&temp,sizeof(float),1,F) == 1);
  }
  assert(fwrite(&dummy,sizeof(int),1,F) == 1);
   
  assert(fwrite(&dummy,sizeof(int),1,F) == 1) ;
  for(i = 0; i < N; i++) {
    temp = si->p[i].r[3];
    assert(fwrite(&temp,sizeof(float),1,F) == 1);
  }
  for(i = 0; i < Nstar; i++) {
    temp = si->pstar[i].r[3];
    assert(fwrite(&temp,sizeof(float),1,F) == 1);
  }
  assert(fwrite(&dummy,sizeof(int),1,F) == 1);
   
  assert(fwrite(&dummy,sizeof(int),1,F) == 1) ;
  for(i = 0; i < N; i++) {
    temp = si->p[i].r[0];
    assert(fwrite(&temp,sizeof(float),1,F) == 1);
  }
  for(i = 0; i < Nstar; i++) {
    temp = si->pstar[i].r[0];
    assert(fwrite(&temp,sizeof(float),1,F) == 1);
  }
  assert(fwrite(&dummy,sizeof(int),1,F) == 1);
   
  assert(fwrite(&dummy,sizeof(int),1,F) == 1) ;
  for(i = 0; i < N; i++) {
    temp = si->p[i].star_flag;
    assert(fwrite(&temp,sizeof(float),1,F) == 1);
  }
  for(i = 0; i < Nstar; i++) {
    temp = si->pstar[i].star_flag;
    assert(fwrite(&temp,sizeof(float),1,F) == 1);
  }
  assert(fwrite(&dummy,sizeof(int),1,F) == 1);
   
}

/*
** Routine for writing density and velocity dispersion profiles for analysis.
*/

void write_profiles(FILE *F, const SI *si,int Ngrid, double rlow, double rup){

  int i,j;
  float dr;
  double rgrid[Ngrid],rho[Ngrid-1],rhoStar[Ngrid-1],vel2[Ngrid-1];
  double vel2star[Ngrid-1],M[Ngrid-1],Mstar[Ngrid-1],r,Msum,MstarSum,vr;
  PARTICLE *p;
  PARTICLE *pstar;

  p = si->p;
  pstar = si->pstar;
  dr = (rup - rlow)/(Ngrid-1.0);
  Msum = 0;
  MstarSum = 0;

  for (j = 0 ; j < Ngrid ; j++){
    if (j < Ngrid -1){
      rho[j] = 0 ;
      rhoStar[j] = 0;
      vel2[j] = 0;
      vel2star[j] = 0;
      M[j] = 0;
      Mstar[j] = 0;
    }
    rgrid[j] = rlow + j*dr;
  }

  if(si->nostarpot_flag == 1){
    for(i = 0 ; i < si->N ; i++){
      for (j = 0 ; j < Ngrid-1 ; j++){
	if ( (p[i].r[0] >= rgrid[j]) && (p[i].r[0] < rgrid[j+1]) ){ 
	  rho[j] += 1;
	  vr = (p[i].v[1]*p[i].r[1] + p[i].v[2]*p[i].r[2] + p[i].v[3]*p[i].r[3])/p[i].r[0];
	  vel2[j] += vr*vr;
	  if (p[i].star_flag > 0) rhoStar[j] += p[i].star_flag;
	}
      }
    }
  }

  if(si->nostarpot_flag == 0){

    for(i = 0 ; i < si->N ; i++){
      for (j = 0 ; j < Ngrid-1 ; j++){
	if ( (p[i].r[0] >= rgrid[j]) && (p[i].r[0] < rgrid[j+1]) ){ 
	  rho[j] += 1;
	  vr = (p[i].v[1]*p[i].r[1] + p[i].v[2]*p[i].r[2] + p[i].v[3]*p[i].r[3])/p[i].r[0];
	  vel2[j] +=  vr*vr;
	}
      }
    }
	
    for(i = 0 ; i < si->Nstar ; i++){
      for (j = 0 ; j < Ngrid-1 ; j++){
	if ( (pstar[i].r[0] >= rgrid[j]) && (pstar[i].r[0] < rgrid[j+1]) ){ 
	  rhoStar[j] += 1;
	  vr = (pstar[i].v[1]*pstar[i].r[1] + pstar[i].v[2]*pstar[i].r[2] + pstar[i].v[3]*pstar[i].r[3])/pstar[i].r[0];
	  vel2star[j] += vr*vr;
	}
      }
    }

  }

  fprintf(F,"#     0             1             2             3             4             5             6 \n");
  fprintf(F,"#     r          rhoHalo       rhoStar       MencHalo      MencStar       vel^2       velStar^2\n");
  for (j = 0 ; j < Ngrid - 1 ; j++){
    r = (rgrid[j] + rgrid[j+1])/2;
    vel2[j] = vel2[j]/rho[j];
    vel2star[j] = vel2star[j]/rhoStar[j];
    Msum += rho[j];
    MstarSum += rhoStar[j];
    M[j] = si->sp->M*Msum/si->N; 
    rho[j] = si->sp->M*rho[j]/(si->N*4*M_PI*r*r*dr);
    if(si->nostarpot_flag == 1){
      rhoStar[j] = si->sp->M*rhoStar[j]/(si->N*4*M_PI*r*r*dr);
      Mstar[j] = si->sp->M*MstarSum/si->N; 
    }else{
      rhoStar[j] = si->sp->Mstar*rhoStar[j]/(si->Nstar*4*M_PI*r*r*dr);
      Mstar[j] = si->sp->Mstar*MstarSum/si->Nstar; 
    }
    fprintf(F, OFD2" "OFD2" "OFD2" "OFD2" "OFD2" "OFD2" "OFD2"\n", r,rho[j],rhoStar[j],M[j],Mstar[j],vel2[j],vel2star[j]);
  }

}

/*
** Routine for writing Gadget binary with initial conditions.
*/


void write_gadget(FILE *fp, const PARTICLE *bh, const SI *si){

  int i, j, dummy, status, Ntotal,Ndark,Nstar;
  float temp;
  PARTICLE *p;
  PARTICLE *pstar;
  GH gh;
    
  p = si->p;
  pstar = si->pstar; 
		
  /*
  **  Assign number of particles.
  */
   
  Nstar = si->Nstar;	
  Ndark = si->N; 
  Ntotal = (si->nostarpot_flag == 0) ? Nstar + Ndark : Ndark;	
  if (bh->mass != 0) {
    Ntotal++;
  }

  /*
  ** Initialise header
  */
  gh.npart[0] = 0;
  gh.npart[1] = Ndark;
  gh.npart[2] = 0;
  gh.npart[3] = 0;
  gh.npart[4] = Nstar;
  gh.npart[5] = 0;
  gh.mass[0] = 0;
  gh.mass[1] = si->mass;
  gh.mass[2] = 0;
  gh.mass[3] = 0;
  gh.mass[4] = si->massStar;
  gh.mass[5] = 0;
  gh.time = 0;
  gh.redshift = 0;
  gh.flag_sfr = 0;
  gh.flag_feedback = 0;
  gh.npartTotal[0] = 0;
  gh.npartTotal[1] = Ndark;
  gh.npartTotal[2] = 0;
  gh.npartTotal[3] = 0;
  gh.npartTotal[4] = Nstar;
  gh.npartTotal[5] = 0;
  gh.flag_cooling = 0;
  gh.num_files = 1;
  gh.BoxSize = 0;
  gh.Omega0 = 0;
  gh.OmegaLambda = 0;
  gh.HubbleParam = 0;
  gh.flag_stellarage = 0;
  gh.flag_metals = 0;
  for (i = 0; i < 6; i++) {
    gh.npartTotalHighWord[i] = 0;
  }
  gh.flag_entropy_instead_u = 0;
  gh.flag_doubleprecision = 0;
  /*
  ** Write out header
  */
  //fprintf(stderr,"Writing gadget header....");
  dummy = sizeof(GH);
  status = fwrite(&dummy,sizeof(int),1,fp);
  assert(status == 1);
  status = fwrite(&gh,sizeof(GH),1,fp);
  assert(status == 1);
  status = fwrite(&dummy,sizeof(int),1,fp);
  assert(status == 1);
  //fprintf(stderr,"done\n");
  /*
  ** Write out positions
  */
  //fprintf(stderr,"Writing gadget positions....");
  dummy = 3*Ntotal*sizeof(float);
  status = fwrite(&dummy,sizeof(int),1,fp);
  assert(status == 1); 
    
  for(i = 0; i < Ndark; i++) {
    if (p[i].star_flag == 0) {
      for(j = 1; j < 4; j++) {
        temp = p[i].r[j];
        status = fwrite(&temp,sizeof(float),1,fp);
        assert(status == 1);
      }
    }
  }
    
  // 	if(si->nostarpot_flag == 1){
  // 		for(i = 0; i < Ntotal; i++) {
  // 		if (p[i].star_flag == 1) {
  // 			for(j = 1; j < 4; j++) {
  // 				temp = p[i].r[j];
  // 				assert(fwrite(&temp,sizeof(float),1,fp) == 1);
  // 				}
  // 			}
  // 		}
  // 	}else{
  for(i = 0; i < Nstar; i++) {
    for(j = 1; j < 4; j++) {
      temp = pstar[i].r[j];
      status = fwrite(&temp,sizeof(float),1,fp);
      assert(status == 1);
    }
		
  }
  //	}		
  //fprintf(stderr,"done\n");
	
  status = fwrite(&dummy,sizeof(int),1,fp);
  assert(status == 1);
    
  /*
  ** Write out velocities
  */
	
  //fprintf(stderr,"Writing gadget velocities....");
  dummy = 3*Ntotal*sizeof(float);
  status = fwrite(&dummy,sizeof(int),1,fp);
  assert(status == 1); 
    		
  for(i = 0; i < Ndark; i++) {
    if (p[i].star_flag == 0) {
      for(j = 1; j < 4; j++) {
        temp = p[i].v[j];
        status = fwrite(&temp,sizeof(float),1,fp);
        assert(status == 1);
      }
    }
  }
	
  //     if(si->nostarpot_flag == 1){
  // 		for(i = 0; i < Ntotal; i++) {
  // 		if (p[i].star_flag == 1) {
  // 			for(j = 1; j < 4; j++) {
  // 				temp = p[i].v[j];
  // 				assert(fwrite(&temp,sizeof(float),1,fp) == 1);
  // 				}
  // 			}
  // 		}
  // 	}else{
  for(i = 0; i < Nstar; i++) {
    for(j = 1; j < 4; j++) {
      temp = pstar[i].v[j];
      status = fwrite(&temp,sizeof(float),1,fp);
      assert(status == 1);
    }		
  }
  //	}
			
  status = fwrite(&dummy,sizeof(int),1,fp);
  assert(status == 1);
  //fprintf(stderr,"done\n");
    
  /*
  ** Write out ids
  */
  //fprintf(stderr,"Writing gadget ids....");
  dummy = Ntotal*sizeof(int);
  status = fwrite(&dummy,sizeof(int),1,fp);
  assert(status == 1);
    
  for(i = 0; i < Ndark; i++) {
    temp = p[i].index;
    status = fwrite(&temp,sizeof(int),1,fp);
    assert(status == 1);
  }
  for(i = 0; i < Nstar; i++) {
    temp = pstar[i].index;
    status = fwrite(&temp,sizeof(int),1,fp);
    assert(status == 1);
  }
  status = fwrite(&dummy,sizeof(int),1,fp);
  assert(status == 1);
  //fprintf(stderr,"done\n");

  //fprintf(stderr,"Finished writing gadget file.\n");
}


/**
 * Write a GIZMO hdf5 file
 *
 * @param fname - Name of hdf5 file
 * @param bh - center BH data
 * @param si - system info
 */
void write_gizmo(char *fname, const PARTICLE *bh, const SI *si){

  int i, j, Ndark,Nstar,Nbh,DMIND,STIND,NUIND;
  PARTICLE *p;
  PARTICLE *pstar;
  GH gh;
  char DMPART[10+1], STPART[10+1], BHPART[10+1];
  char field[30];

  // Dark and star particle arrays
  p = si->p;
  pstar = si->pstar;

  /*
  **  Assign number of particles.
  */
  Nstar = si->Nstar;
  Ndark = si->N;
  // We don't need Ntotal, but if I'm understanding correctly, this actually
  // effectively turns off stars if nostarpot_flag is true. I don't think
  // that's what we want for gizmo. It'd be better to do that in the config
  // or parameter files.
  //Ntotal = (si->nostarpot_flag == 0) ? Nstar + Ndark : Ndark;
  // The write_gadget function just includes the central BH by incrementing
  // Ntotal and does nothing else. This is insufficient in general but will
  // only be fixed here
  if (bh->mass != 0) {
    Nbh = 1;
  } else {
    Nbh = 0;
  }

  /*
  ** In the future, we should add a flag to specify that we want to switch 
  ** particle types, or even some sort of mapping. For now we will assume
  ** that if -starabg is given, then PartType1 represents high resolution
  ** inner particles (aka the star distribution) and PartType2 represents
  ** low-resolution outer particles (aka the DM distribution). Otherwise,
  ** PartType1 represents the DM distribution and PartType4 represents the
  ** stellar distribution. PartType5 is always BHs
  */
  strcpy(BHPART,"/PartType5");
  // For a still unknown reason, the spheric-gizmo pipeline has a _lot_ of 
  // trouble with putting the HR particles as PartType1 and LR particles as
  // PartType2. So we'll leave things as originally designed
  //if (si->starabg_flag == 1) {
  //  strcpy(DMPART,"/PartType2");
  //  strcpy(STPART,"/PartType1");
  //  DMIND = 2;
  //  STIND = 1;
  //  NUIND = 4;
  //} else {
    strcpy(DMPART,"/PartType1");
    strcpy(STPART,"/PartType4");
    DMIND = 1;
    STIND = 4;
    NUIND = 2;
  //}

  /*
  ** Initialise header - we can reuse most of this since GIZMO expects
  ** backwards compatability.
  */
  gh.npart[0] = 0;
  gh.npart[DMIND] = Ndark;
  gh.npart[NUIND] = 0;
  gh.npart[3] = 0;
  gh.npart[STIND] = Nstar;
  gh.npart[5] = Nbh;
  gh.mass[0] = 0;
  gh.mass[DMIND] = si->mass;
  gh.mass[NUIND] = 0;
  gh.mass[3] = 0;
  gh.mass[STIND] = si->massStar;
  gh.mass[5] = bh->mass;
  gh.npartTotal[0] = 0;
  gh.npartTotal[DMIND] = Ndark;
  gh.npartTotal[NUIND] = 0;
  gh.npartTotal[3] = 0;
  gh.npartTotal[STIND] = Nstar;
  gh.npartTotal[5] = Nbh;
  for (i = 0; i < 6; i++) {
    gh.npartTotalHighWord[i] = 0;
  }
  gh.time = 0;
  gh.num_files = 1;
  gh.flag_doubleprecision = 1;
  gh.BoxSize = 0; // This is ignored by gizmo but used by other software
  // The following are ignored as per gizmo-public/scripts/make_IC.py.
  // Most of them are set in the params or Config file
  //gh.redshift = 0;
  //gh.flag_sfr = 0;
  //gh.flag_feedback = 0;
  //gh.flag_cooling = 0;
  //gh.Omega0 = 0;
  //gh.OmegaLambda = 0;
  //gh.HubbleParam = 0;
  //gh.flag_stellarage = 0;
  //gh.flag_metals = 0;
  //gh.flag_entropy_instead_u = 0;

  /* now the various steps involved in preparing an hdf5 file */
  hid_t file_id, grp;
  /* create a HDF5 file */
  file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  // Only need to create groups for PartType1, PartType2/PartType4, and PartType5
  herr_t status;


  ////////  Do dark particles
  fprintf(stderr,"Saving dark particles\nCreating and closing group....");
  grp = H5Gcreate(file_id, DMPART, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Gclose(grp);
  fprintf(stderr,"closed.\nCreating particles....");
  assert(status >= 0);
  hsize_t dims[2] = {Ndark, 3};
  double *ddrk = malloc(Ndark*3*sizeof(double));
  // Do coordinates
  for(i=0;i<Ndark;i++){
    // Need to initialize arrays first
    for(j=0;j<3;j++){
      ddrk[i*3+j] = p[i].r[j+1];
    }
  }

  /* create and write a double type dataset named "DMPART/Coordinates" */
  fprintf(stderr,"created.\nSaving coordinates dataset....");
  snprintf(field, sizeof(field), "%s%s", DMPART, "/Coordinates");
  status = H5LTmake_dataset(file_id, field , 2, dims,
                            H5T_NATIVE_DOUBLE, ddrk);
  assert(status != -1);
  fprintf(stderr,"saved.\nSaving velocities dataset....");
  // do velocities - since velocities are also an Ndark x 3 array, reuse ddrk
  for(i=0;i<Ndark;i++){
    for(j=0;j<3;j++){
      ddrk[i*3 + j] = p[i].v[j+1];
    }
  }
  snprintf(field, sizeof(field), "%s%s", DMPART, "/Velocities");
  status = H5LTmake_dataset(file_id, field , 2, dims,
                            H5T_NATIVE_DOUBLE, ddrk);
  assert(status != -1);
  free(ddrk);
  fprintf(stderr,"saved.\nSaving particle ids....");
  // do number id
  dims[1] = 1; // Now only a Ndark x 1 array
  double *adrk = malloc(Ndark*sizeof(double));
  for(i=0;i<Ndark;i++){
    adrk[i] = p[i].index;
  }
  snprintf(field, sizeof(field), "%s%s", DMPART, "/ParticleIDs");
  status = H5LTmake_dataset(file_id, field , 1, dims,
                            H5T_NATIVE_DOUBLE, adrk);
  assert(status != -1);
  fprintf(stderr,"saved.\nSaving masses....");
  // do masses - can reuse adrk
  for(i=0;i<Ndark;i++){
    adrk[i] = gh.mass[DMIND];
  }
  snprintf(field, sizeof(field), "%s%s", DMPART, "/Masses");
  status = H5LTmake_dataset(file_id, field, 1, dims,
                            H5T_NATIVE_DOUBLE, adrk);
  assert(status != -1);
  free(adrk);
  fprintf(stderr,"saved.\n");

  /////// Do star particles

  if(Nstar > 0){
    fprintf(stderr,"Saving star particles\nCreating and closing group....");
    grp = H5Gcreate(file_id, STPART, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Gclose(grp);
    assert(status >= 0);
    fprintf(stderr,"closed.\nCreating particles....");
    dims[0] = Nstar;
    dims[1] = 3;
    double *dstr = malloc(Nstar*3*sizeof(double));
    // Do coordinates
    for(i=0;i<Nstar;i++){
      // Need to initialize arrays first
      for(j=0;j<3;j++){
        dstr[i*3 + j] = pstar[i].r[j+1];
      }
    }

    fprintf(stderr,"created.\nSaving coordinates dataset....");
    /* create and write a double type dataset named "STPART/Coordinates" */
    snprintf(field, sizeof(field), "%s%s", STPART, "/Coordinates");
    status = H5LTmake_dataset(file_id, field, 2, dims,
                            H5T_NATIVE_DOUBLE, dstr);
    assert(status != -1);
    fprintf(stderr,"saved.\nSaving velocities dataset....");
    // do velocities - since velocities are also an Nstar x 3 array, reuse dstr
    for(i=0;i<Nstar;i++){
      for(j=0;j<3;j++){
        dstr[i*3 + j] = pstar[i].v[j+1];
      }
    }
    snprintf(field, sizeof(field), "%s%s", STPART, "/Velocities");
    status = H5LTmake_dataset(file_id, field, 2, dims,
                              H5T_NATIVE_DOUBLE, dstr);
    assert(status != -1);
    free(dstr);
    fprintf(stderr,"saved.\nSaving particle ids....");
    // do number id - needs to start from Ndark
    dims[1] = 1; // Now only a Nstar x 1 array
    double *astr = malloc(Nstar*sizeof(double));
    for(i=0;i<Nstar;i++){
      astr[i] = pstar[i].index;
    }
    snprintf(field, sizeof(field), "%s%s", STPART, "/ParticleIDs");
    status = H5LTmake_dataset(file_id, field, 1, dims,
                              H5T_NATIVE_DOUBLE, astr);
    assert(status != -1);
    fprintf(stderr,"saved.\nSaving masses....");
    // do masses - can reuse adrk
    for(i=0;i<Nstar;i++){
      astr[i] = gh.mass[STIND];
    }
    snprintf(field, sizeof(field), "%s%s", STPART, "/Masses");
    status = H5LTmake_dataset(file_id, field, 1, dims,
                              H5T_NATIVE_DOUBLE, astr);
    assert(status != -1);
    free(astr);
    fprintf(stderr,"saved.\n");
  }

  /////// Do black holes
  if(Nbh > 0){
    fprintf(stderr,"Saving BH particles\nCreating and closing group....");
    grp = H5Gcreate(file_id, BHPART, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Gclose(grp);
    fprintf(stderr,"closed.\nCreating particles....");
    assert(status >= 0);
    // Leaving dbh on the stack since Nbh shoud be only 1 (or at least small)
    double dbh[Nbh][3];
    dims[0] = Nbh;
    dims[1] = 3;
    // Do coordinates
    for(i=0;i<Nbh;i++){
      for(j=0;j<3;j++){
        dbh[i][j] = bh->r[j+1];
      }
    }
    fprintf(stderr,"created.\nSaving coordinates dataset....");
    /* create and write a double type dataset named "/PartType5/Coordinates" */
    snprintf(field, sizeof(field), "%s%s", BHPART, "/Coordinates");
    status = H5LTmake_dataset(file_id, field, 2, dims,
                            H5T_NATIVE_DOUBLE, dbh);
    assert(status != -1);
    fprintf(stderr,"saved.\nSaving velocities dataset....");
    // do velocities - since velocities are also an Nbh x 3 array, reuse dbh
    for(i=0;i<Nbh;i++){
      for(j=0;j<3;j++){
        dbh[i][j] = bh->v[j+1];
      }
    }
    snprintf(field, sizeof(field), "%s%s", BHPART, "/Velocities");
    status = H5LTmake_dataset(file_id, field, 2, dims,
                            H5T_NATIVE_DOUBLE, dbh);
    assert(status != -1);
    fprintf(stderr,"saved.\nSaving particle ids....");
    // do number id - needs to start from Ndark + Nstar + 1
    dims[1] = 1; // Now only a Nbh x 1 array
    double abh[Nbh];
    for(i=0;i<Nbh;i++){
      abh[i] = i + Ndark + Nstar + 1;
    }
    snprintf(field, sizeof(field), "%s%s", BHPART, "/ParticleIDs");
    status = H5LTmake_dataset(file_id, field, 1, dims,
                            H5T_NATIVE_DOUBLE, abh);
    assert(status != -1);
    fprintf(stderr,"saved.\nSaving masses....");
    // do masses - can reuse adrk
    for(i=0;i<Nbh;i++){
      abh[i] = bh->mass;
    }
    snprintf(field, sizeof(field), "%s%s", BHPART, "/Masses");
    status = H5LTmake_dataset(file_id, field, 1, dims,
                            H5T_NATIVE_DOUBLE, abh);
    assert(status != -1);
    fprintf(stderr,"saved.\n");
  }

  /* add some hdf5 attributes with the metadata we need */
  fprintf(stderr,"Saving Header.\nCreating and closing group....");
  grp = H5Gcreate(file_id, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Gclose(grp);
  assert(status >= 0);
  fprintf(stderr,"closed.\nSaving header fields....");
  // here we set all the basic numbers that go into the header
  // (most of these will be written over anyways if it's an IC file; the only thing we actually *need* to be 'correct' is "npart")
  H5LTset_attribute_uint(file_id, "/Header", "NumPart_ThisFile", gh.npart, 6);
  H5LTset_attribute_int(file_id, "/Header", "NumPart_Total", gh.npartTotal, 6);
  H5LTset_attribute_int(file_id, "/Header", "NumPart_Total_HighWord", gh.npartTotalHighWord, 6); //This should get overwritten I think?
  // Since we won't assume particles will have constant masses, we will set
  // this to an array of 0s instead of using gh.mass
  float massTable[6] = {0,0,0,0,0,0};
  H5LTset_attribute_float(file_id, "/Header", "MassTable", massTable, 6);
  H5LTset_attribute_double(file_id, "/Header", "Time", &gh.time, 1);
  H5LTset_attribute_int(file_id, "/Header", "NumFilesPerSnapshot", &gh.num_files, 1);
  H5LTset_attribute_int(file_id, "/Header", "Flag_DoublePrecision", &gh.flag_doubleprecision, 1);
  char *gizver = "2023";
  H5LTset_attribute_char(file_id, "/Header", "GIZMO_version", gizver, 1);
  H5LTset_attribute_double(file_id, "/Header", "BoxSize", &gh.BoxSize, 1);
  fprintf(stderr,"saved.\nClosing file....");
  status = H5Fclose (file_id); assert(status != -1);
  assert(status != -1);
  H5close();
  fprintf(stderr,"closed.\n");
}

/*
** Routine for writing ASCII files with initial conditions.
*/

void write_ics(FILE *F, const SI *si,int flag){

  int i,j;

  //     if (bh->mass != 0) {
  // 	assert(fprintf(file,OFI1" ",index) > 0);
  //         assert(fprintf(file,OFD5" ",bh->mass) > 0);
  // 	for (j = 0; j < 3; j++) {
  // 	    assert(fprintf(file,OFD6" ",bh->r[j+1]) > 0);
  //             }
  //         for (j = 0; j < 3; j++) {
  //             assert(fprintf(file,OFD6" ",bh->v[j+1]) > 0);
  //             }
  // 	assert(fprintf(file,"\n") > 0);
  // 	index++;
  //         }
	
  if (flag == 1) {
    fprintf(F,"#0       1              2              3              4              5              6              7              8\n");
    fprintf(F,"#n   haloPmass        haloX          haloY          haloZ          haloVx         haloVy        haloVz      f(E)star/f(E)\n");
    for (i = 0; i < si->N; i++) {
      assert(fprintf(F,OFI1" ",si->p[i].index) > 0);
      assert(fprintf(F,OFD3" ",si->p[i].mass) > 0);
      for (j = 0; j < 3; j++) {
	assert(fprintf(F,OFD3" ",si->p[i].r[j+1]) > 0);
      }
      for (j = 0; j < 3; j++) {
	assert(fprintf(F,OFD3" ",si->p[i].v[j+1]) > 0);
      }
      assert(fprintf(F,OFD3" ",si->p[i].star_flag) > 0);
      assert(fprintf(F,"\n") > 0);
    }
  }
	
  if (flag == 2) {
    fprintf(F,"#0       1             2              3              4              5              6              7\n");
    fprintf(F,"#n   starPmass       starX          starY          starZ          starVx         starVy         starVz\n");		
    for (i = 0; i < si->Nstar; i++) {
      assert(fprintf(F,OFI1" ",si->pstar[i].index) > 0);
      assert(fprintf(F,OFD3" ",si->pstar[i].mass) > 0);
      for (j = 0; j < 3; j++) {
	assert(fprintf(F,OFD3" ",si->pstar[i].r[j+1]) > 0);
      }
      for (j = 0; j < 3; j++) {
	assert(fprintf(F,OFD3" ",si->pstar[i].v[j+1]) > 0);
      }	
      assert(fprintf(F,"\n") > 0);
		
    }
  }
}

/*
** Routine for writing typsy binary with initial conditions.
*/

void write_tipsy(FILE *fp, const TIPSY_STRUCTURE *ts) {

  TIPSY_HEADER *th;
  GAS_PARTICLE *gp;
  DARK_PARTICLE *dp;
  STAR_PARTICLE *sp;

  th = ts->th;
  gp = ts->gp;
  dp = ts->dp;
  sp = ts->sp;
  /*
  ** Write out header
  */
  assert(fwrite(th,sizeof(TIPSY_HEADER),1,fp) == 1);
  /*
  ** Write out particles
  */
  assert(fwrite(gp,sizeof(GAS_PARTICLE),th->ngas,fp) == th->ngas);
  assert(fwrite(dp,sizeof(DARK_PARTICLE),th->ndark,fp) == th->ndark);
  assert(fwrite(sp,sizeof(STAR_PARTICLE),th->nstar,fp) == th->nstar);
}
