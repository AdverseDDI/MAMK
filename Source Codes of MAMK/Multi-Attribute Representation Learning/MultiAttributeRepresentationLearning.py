from matplotlib import pyplot     
import cupy as np
import numpy as npp
import pandas as pd
from matplotlib import pylab  
from sklearn.datasets import load_files   
from sklearn.feature_extraction.text import  CountVectorizer  
from sklearn.feature_extraction.text import  TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import precision_recall_curve, roc_curve, auc  
from sklearn.metrics import classification_report  
from sklearn.linear_model import LogisticRegression  
import time 
# from scipy.linalg.misc import norm
from numpy import *
import os
np.set_printoptions(threshold=np.inf) 
from sklearn.metrics import roc_auc_score, roc_curve, auc 
from sklearn import metrics 
import time
from sklearn.metrics import classification_report, average_precision_score
from sklearn.linear_model import LogisticRegression 
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(1)
torch.manual_seed(1)


def LoadDrugAttributeMatrix(DrugAttributeAddress):
	DrugAttributeMatrix=[]
	fileIn=open(DrugAttributeAddress)
	line=fileIn.readline()
	while line:
		lineArr=line.strip().split('\t')
		# Split each line by Tab i.e., \t
		temp=[]
		if len(lineArr)==1:
			DrugAttributeMatrix.append(temp)
			# append the Molecular Structure of each drug (i.e., temp) to DrugStructureMatrix
			line=fileIn.readline()
			continue
		lineSpace=lineArr[1].strip().split()
		# The first entry is the drug Name; The second entry is  
		# the feature vector for the attrubute information of drugs
		# we use the second entry to get the attribute Matrix
		for i in lineSpace:
			temp.append(int(i))
			# each element is appended into array temp to get the Molecular Structure vector of each drug
		# temp=np.array(temp)
		DrugAttributeMatrix.append(temp)
		# append the Molecular Structure of each drug (i.e., temp) to DrugStructureMatrix
		line=fileIn.readline()

	return DrugAttributeMatrix	

def GetExistingMissingDrugAttributeMatrixIndex(DrugAttributeMatrix):
	# return the existing and missing rows i.e., the drugs with 
	# attribute information and no attribute information 
	DrugNum=len(DrugAttributeMatrix)
	DrugIndexWithAttribute=[]
	DrugIndexWithNoAttribute=[]
	for Index in range(DrugNum):
		if DrugAttributeMatrix[Index]==[]:
			DrugIndexWithNoAttribute.append(Index)
		else:
			DrugIndexWithAttribute.append(Index)
	return DrugIndexWithAttribute, DrugIndexWithNoAttribute

def GetExistingDrugAttributeMatrixwithIndicatingMatrix(DrugAttributeMatrix,AttributeDrugIndex,NoAttributeDrugIndex):
	# this function is to return the drug attribute matrix with the existing attribute information
	# and also return the indicating matrix that can convert the original drug attribute matrix into 
	# the existing drug attribute matrix
	DrugNum=len(DrugAttributeMatrix)
	DrugExistingMatrix=[]
	for Index in AttributeDrugIndex:
		# traverse the index with the known drug attribute information 
		DrugExistingMatrix.append(DrugAttributeMatrix[Index])
	DrugExistingMatrix=np.array(DrugExistingMatrix)

	H_ExistingIndicatingMatrix=np.eye(DrugNum)
	H_ExistingIndicatingMatrix=np.asnumpy(H_ExistingIndicatingMatrix)
	H_ExistingIndicatingMatrix=npp.delete(H_ExistingIndicatingMatrix,NoAttributeDrugIndex,axis=0)
	H_ExistingIndicatingMatrix=np.array(H_ExistingIndicatingMatrix)
	# print(H_ExistingIndicatingMatrix)
	H_MissingIndicatingMatrix=np.eye(DrugNum)
	H_MissingIndicatingMatrix=np.asnumpy(H_MissingIndicatingMatrix)
	H_MissingIndicatingMatrix=npp.delete(H_MissingIndicatingMatrix,AttributeDrugIndex,axis=0)
	H_MissingIndicatingMatrix=np.array(H_MissingIndicatingMatrix)
	return DrugExistingMatrix,H_ExistingIndicatingMatrix,H_MissingIndicatingMatrix

def GetExistingMissingRecoveryIndicatingMatrix(AttributeDrugIndex,NoAttributeDrugIndex):
# this function is to get the recovery indicating matrix to recover the original drug attribute matrix
# by the existing drug attribute matrix and missing drug attribute matrix
	ExistingDrugNum=len(AttributeDrugIndex)
	MissingDrugNum=len(NoAttributeDrugIndex)
	DrugNum=ExistingDrugNum+MissingDrugNum
	ExistingRecoveryIndicatingMatrix=np.zeros((DrugNum,ExistingDrugNum))
	MissingRecoveryIndicatingMatrix=np.zeros((DrugNum,MissingDrugNum))
	Count=0
	for Index in AttributeDrugIndex:
		ExistingRecoveryIndicatingMatrix[Index,Count]=1
		Count=Count+1
	Count=0
	for Index in NoAttributeDrugIndex:
		MissingRecoveryIndicatingMatrix[Index,Count]=1
		Count=Count+1
	return ExistingRecoveryIndicatingMatrix,MissingRecoveryIndicatingMatrix


def ReturnDiagonalMatrixofNoSquareMatrix(X_Dimension,Y_Dimension):
	# this function is to return a no square matrix with size of 
	# X_Dimension*Y_Dimensionthat and the main diagonal elements is 1
	# and the remaining elements is 0.  
	LeastDimension=min(X_Dimension,Y_Dimension)
	NoSquareMatrix=np.zeros((X_Dimension,Y_Dimension))
	for Index in range(LeastDimension):
		NoSquareMatrix[Index,Index]=1
	return NoSquareMatrix

def ReturnDiagonalMatrixofOnePowerandThreePower(InputMatrix_X):
	# this function is to return two Diagonal matrices, the Diagonal elements 
	# of the first one are inverse of the L_2-norm of each row of InputMatrix_X; and the Diagonal elements 
	# of the second one are inverse of the three power of the L_2 norm of each row of InputMatrix_X
	OnePowerInvertVector=np.linalg.norm(InputMatrix_X,ord=None,axis=1)
	OnePowerInvertVector=1/OnePowerInvertVector
	OnePowerInvertMatrix=np.diag(OnePowerInvertVector)
	ThreePowerInvertMatrix=OnePowerInvertMatrix*OnePowerInvertMatrix*OnePowerInvertMatrix
	return OnePowerInvertMatrix,ThreePowerInvertMatrix

def UpdateX_AttributeFunction(X_Attribute,P_Matrix,Q_Matrix,U_Matrix,Beta):
	# this function is to update X_attribute
	OnePowerInvertMatrix,ThreePowerInvertMatrix=ReturnDiagonalMatrixofOnePowerandThreePower(X_Attribute)
	DrugNum=np.shape(P_Matrix)[0]
	TempMatrix=np.dot(P_Matrix+Q_Matrix,(P_Matrix+Q_Matrix).transpose())
	TempMatrixandItsTransposeDiag=np.diag(TempMatrix) # a vector
	FirstMatrix=np.zeros((DrugNum,DrugNum));SecondMatrix=np.zeros((DrugNum,DrugNum))
	FirstMatrix[:,0:DrugNum]=TempMatrixandItsTransposeDiag.reshape(DrugNum,1)
	SecondMatrix[0:DrugNum,:]=TempMatrixandItsTransposeDiag.reshape(1,DrugNum)
	E_Matrix=FirstMatrix+SecondMatrix-2*TempMatrix
	# OriginalDiagMatrix=X_Attribute*X_Attribute.transpose()*OnePowerInvertMatrix*E_Matrix
	OriginalDiagMatrix=np.dot(np.dot(np.dot(X_Attribute,X_Attribute.transpose()),OnePowerInvertMatrix),E_Matrix)
	DiagonalMatrix=np.diag(np.diag(OriginalDiagMatrix))
	
	Numerator=np.dot(P_Matrix+Q_Matrix,U_Matrix)+Beta*np.dot(np.dot(DiagonalMatrix,ThreePowerInvertMatrix),X_Attribute)
	Denominator=X_Attribute+Beta*np.dot(np.dot(np.dot(OnePowerInvertMatrix,E_Matrix),OnePowerInvertMatrix),X_Attribute)
	TempValue=X_Attribute*Numerator

	X_Attribute=TempValue/Denominator
	X_Attribute=np.where(Denominator!=0,X_Attribute,np.zeros_like(X_Attribute))
	# filt the elements with Denominator=0
	# print(X_Attribute)
	return X_Attribute


def ReturnStandardConsinofX_AttributeMatrix(X_Attribute):
	# this function is to get the standard cosine matrix of X_Attribute
	DrugNum=np.shape(X_Attribute)[0]
	L_2_norm=np.linalg.norm(X_Attribute,ord=None,axis=1)
	ColumnL_2_norm=L_2_norm.reshape(DrugNum,1)
	RowL_2_norm=L_2_norm.reshape(1,DrugNum)
	ColumnRow=np.dot(ColumnL_2_norm,RowL_2_norm)
	F_Matrix=np.dot(X_Attribute,X_Attribute.transpose())/ColumnRow
	F_Matrix=np.where(ColumnRow!=0,ColumnRow,np.zeros_like(ColumnRow))
	return F_Matrix

def EigenDecompositionX_Attribute(X_Attribute):
	# this function is to return the left and right matrices and Eigenvalue matrix of X_Attribute
	X_Attribute=np.asnumpy(X_Attribute)
	EigenVector,DecompositionMatrix=npp.linalg.eig(X_Attribute)
	EigenVector=EigenVector.reshape((1,np.shape(X_Attribute)[0]))
	EigenVector=np.array(EigenVector)
	DecompositionMatrix=np.array(DecompositionMatrix)
	return EigenVector,DecompositionMatrix

def CalculateDiagonalMatrixofSumofRow(F_Matrix):
	# this function is to sum up the each row of F_Matrix and return a diagonal matrix
	# with the sums being the diagonal elements
	DiagonalMatrix=np.diag(np.sum(F_Matrix,axis=1))
	return DiagonalMatrix

def CalculateNumeratorandDenominatorforUpdatingP(List_X_Attribute,P_CommonAttributeMatrix,
	List_Q_Matrix,List_U_Matrix,List_F_Attribute,List_DiagonalF_Attribute,ListBeta):
	AttributeNum=len(ListBeta)
	NumeratorforUpdatingP=np.zeros((np.shape(P_CommonAttributeMatrix)))
	DenominatorforUpdatingP=np.zeros((np.shape(P_CommonAttributeMatrix)))
	for Attribute_Index in range(AttributeNum):
		X_Attribute=List_X_Attribute[Attribute_Index]
		Q_Matrix=List_Q_Matrix[Attribute_Index]
		U_Matrix=List_U_Matrix[Attribute_Index]
		Beta=ListBeta[Attribute_Index]
		F_Matrix=List_F_Attribute[Attribute_Index]
		DiagonalF_Matrix=List_DiagonalF_Attribute[Attribute_Index]
		NumeratorforUpdatingP=NumeratorforUpdatingP+np.dot(X_Attribute,U_Matrix.transpose())+\
			Beta*np.dot(F_Matrix,P_CommonAttributeMatrix+Q_Matrix)
		DenominatorforUpdatingP=DenominatorforUpdatingP+np.dot(P_CommonAttributeMatrix+Q_Matrix,np.dot(U_Matrix,U_Matrix.transpose()))+\
			Beta*np.dot(DiagonalF_Matrix,P_CommonAttributeMatrix+Q_Matrix)
	return NumeratorforUpdatingP,DenominatorforUpdatingP


def CalculateBetaAndDeltaMatrix(ListBeta,ListDeltaMatrix):
	AttributeNum=len(ListBeta)
	DrugNum=np.shape(ListDeltaMatrix[0])[0]
	All_Attribute=np.zeros((DrugNum,DrugNum))
	for Attribute_Index in range(AttributeNum):
		Beta=ListBeta[Attribute_Index]
		DeltaMatrix=ListDeltaMatrix[Attribute_Index]
		# print(type(Beta),type(DeltaMatrix))
		All_Attribute=All_Attribute+Beta*DeltaMatrix
	return All_Attribute

def CalculateU_MatrixandTranspose(List_U_Matrix):
	# this function is to calculate all attribute U_Matrix*U_Matrix.transpose()
	AttributeNum=len(List_U_Matrix)
	Common_Dimention=np.shape(List_U_Matrix[0])[0]
	All_Attribute=np.zeros((Common_Dimention,Common_Dimention))
	for Attribute_Index in range(AttributeNum):
		TempMatrix=List_U_Matrix[Attribute_Index]
		All_Attribute=All_Attribute+np.dot(TempMatrix,TempMatrix.transpose())
	return All_Attribute

def CalculateRightC_mForUpdateMatrixP(List_X_Attribute,List_Q_Matrix,List_U_Matrix,ListDeltaMatrix,ListBeta):
	# this function is to calculate the right equation for updating matrix P
	AttributeNum=len(ListBeta)
	DrugNum=np.shape(List_X_Attribute[0])[0]
	Common_Dimention=np.shape(List_U_Matrix[0])[0]
	All_Attribute=np.zeros((DrugNum,Common_Dimention))
	for Attribute_Index in range(AttributeNum):
		FirstTem=np.dot(List_X_Attribute[Attribute_Index],List_U_Matrix[Attribute_Index].transpose())
		SecondTem=np.dot(np.dot(List_Q_Matrix[Attribute_Index],List_U_Matrix[Attribute_Index]),List_U_Matrix[Attribute_Index].transpose())
		ThirdTem=ListBeta[Attribute_Index]*np.dot(ListDeltaMatrix[Attribute_Index],List_Q_Matrix[Attribute_Index])
		All_Attribute=All_Attribute+FirstTem-SecondTem-ThirdTem
	return All_Attribute

def UpdateMatrixQ_Attribute(X_Attribute,U_Matrix,P_CommonAttributeMatrix,F_Matrix,
	DiagonalF_Matrix,Q_Matrix,List_Q_Matrix,gamma,beta):
	DrugNum,Common_Dimention=np.shape(Q_Matrix)
	AttributeNum=len(List_Q_Matrix)
	ALL_Ones=np.ones((DrugNum,Common_Dimention))*gamma*AttributeNum
	Numerator=np.dot(X_Attribute,U_Matrix.transpose())+beta*np.dot(F_Matrix,(P_CommonAttributeMatrix+Q_Matrix))
	DivisionMatrix=np.zeros((DrugNum,Common_Dimention))
	DivisionMatrixPositive=np.zeros((DrugNum,Common_Dimention))
	DivisionMatrixNegative=np.zeros((DrugNum,Common_Dimention))
	for Attribute in range(AttributeNum):
		Division=List_Q_Matrix[Attribute]/Q_Matrix
		Division=np.where(Q_Matrix!=0,Division,np.zeros_like(Division))
		Division=np.where(List_Q_Matrix[Attribute]!=0,Division,np.zeros_like(Division))
		DivisionMatrix=DivisionMatrix+Division

		DivisionReverse=1/Division
		DivisionReverselog=np.log(DivisionReverse)
		DivisionReverselog=np.where(DivisionReverse!=0,DivisionReverselog,
			np.zeros_like(DivisionReverselog))

		DivisionReverselogPositive=0.5*(np.absolute(DivisionReverselog)+DivisionReverselog)
		DivisionReverselogNegative=0.5*(np.absolute(DivisionReverselog)-DivisionReverselog)
		DivisionMatrixPositive=DivisionMatrixPositive+DivisionReverselogPositive
		DivisionMatrixNegative=DivisionMatrixNegative+DivisionReverselogNegative
	Numerator=Numerator+ALL_Ones+gamma*DivisionMatrixPositive
	Denominator=np.dot(np.dot(P_CommonAttributeMatrix+Q_Matrix,U_Matrix),U_Matrix.transpose())+\
		beta*np.dot(DiagonalF_Matrix,P_CommonAttributeMatrix+Q_Matrix)+gamma*DivisionMatrix+gamma*DivisionMatrixNegative
	Q_Matrix=Q_Matrix*Numerator/Denominator
	Q_Matrix=np.where(Denominator!=0,Q_Matrix,np.zeros_like(Q_Matrix))
	WhereIsNan=np.isnan(Q_Matrix)
	Q_Matrix[WhereIsNan]=0.0 
	return Q_Matrix

def UpdateMatrixHatU_Attribute(X_Attribute,P_CommonAttributeMatrix,Q_Matrix,Rou,Lambda_Matrix,U_Matrix):
	# this function is to update HatU_Matrix for each attribute
	Common_Dimention=np.shape(P_CommonAttributeMatrix)[1]
	TempMatrix=P_CommonAttributeMatrix+Q_Matrix
	Identity=np.eye(Common_Dimention)
	LeftMatrix=2*np.dot(TempMatrix.transpose(),TempMatrix)+Rou*Identity
	LeftMatrix=np.linalg.inv(LeftMatrix)
	RightMatrix=2*np.dot(TempMatrix.transpose(),X_Attribute)+Rou*(U_Matrix+Lambda_Matrix/Rou)
	return np.maximum(np.dot(LeftMatrix,RightMatrix),0)

def UpdateMatrixHatU_Attribute_Updation2(X_Attribute,P_CommonAttributeMatrix,Q_Matrix,Rou,Lambda_Matrix,U_Matrix,Hat_U_Matrix):
	TempMatrix=P_CommonAttributeMatrix+Q_Matrix
	Numerator=2*np.dot(TempMatrix.transpose(),X_Attribute)+Rou*U_Matrix+Lambda_Matrix
	Denominator=2*np.dot(np.dot(TempMatrix.transpose(),TempMatrix),Hat_U_Matrix)+Rou*Hat_U_Matrix
	Hat_U_Matrix=Hat_U_Matrix*Numerator/Denominator
	Hat_U_Matrix=np.where(Denominator!=0,Hat_U_Matrix,np.zeros_like(Hat_U_Matrix))
	return Hat_U_Matrix

def UpdateMatrixU_Attribute(Alpha,Hat_U_Matrix,Lambda_Matrix,Rou):
	# this function is to updata matrix U for each attribute 
	U_Matrix=np.zeros((np.shape(Hat_U_Matrix)))
	RouLambda_Matrix=Hat_U_Matrix-Lambda_Matrix/Rou
	AlphaRou=Alpha/Rou
	U_Matrix1=np.where(RouLambda_Matrix>=AlphaRou,RouLambda_Matrix-AlphaRou,np.zeros_like(U_Matrix))
	U_Matrix2=np.where(RouLambda_Matrix<=-AlphaRou,RouLambda_Matrix+AlphaRou,np.zeros_like(U_Matrix))
	U_Matrix=U_Matrix2+U_Matrix1
	return U_Matrix


def LearningProcessforUpdateParameters(DrugMolecularStructureExistingMatrix,H_ExistingIndicatingMolecularStructureMatrix,
	H_MissingIndicatingMolecularStructureMatrix,ExistingRecoveryIndicatingMolecularStructureMatrix,MissingRecoveryIndicatingMolecularStructureMatrix,
	DrugTargetExistingMatrix,H_ExistingIndicatingTargetMatrix,H_MissingIndicatingTargetMatrix,
	ExistingRecoveryIndicatingTargetMatrix,MissingRecoveryIndicatingTargetMatrix,
	DrugPathwayExistingMatrix,H_ExistingIndicatingPathwayMatrix,H_MissingIndicatingPathwayMatrix,
	ExistingRecoveryIndicatingPathwayMatrix,MissingRecoveryIndicatingPathwayMatrix,
	DrugSideEffectExistingMatrix,H_ExistingIndicatingSideEffectMatrix,H_MissingIndicatingSideEffectMatrix,
	ExistingRecoveryIndicatingSideEffectMatrix,MissingRecoveryIndicatingSideEffectMatrix,
	DrugPhenotypeExistingMatrix,H_ExistingIndicatingPhenotypeMatrix,H_MissingIndicatingPhenotypeMatrix,
	ExistingRecoveryIndicatingPhenotypeMatrix,MissingRecoveryIndicatingPhenotypeMatrix,
	DrugGeneExistingMatrix,H_ExistingIndicatingGeneMatrix,H_MissingIndicatingGeneMatrix,
	ExistingRecoveryIndicatingGeneMatrix,MissingRecoveryIndicatingGeneMatrix,
	DrugDiseaseExistingMatrix,H_ExistingIndicatinDiseaseMatrix,H_MissingIndicatingDiseaseMatrix,
	ExistingRecoveryIndicatingDiseaseMatrix,MissingRecoveryIndicatingDiseaseMatrix):
	DrugNum=np.shape(H_ExistingIndicatingMolecularStructureMatrix)[0]+\
		np.shape(H_MissingIndicatingMolecularStructureMatrix)[0]

	Dimension_MolecularStructure=np.shape(DrugMolecularStructureExistingMatrix)[1]
	Dimension_Target=np.shape(DrugTargetExistingMatrix)[1]
	Dimension_Pathway=np.shape(DrugPathwayExistingMatrix)[1]
	Dimension_SideEffect=np.shape(DrugSideEffectExistingMatrix)[1]
	Dimension_Phenotype=np.shape(DrugPhenotypeExistingMatrix)[1]
	Dimension_Gene=np.shape(DrugGeneExistingMatrix)[1]
	Dimension_Disease=np.shape(DrugDiseaseExistingMatrix)[1]

	Dimension_Range1=max(Dimension_MolecularStructure,Dimension_Target,Dimension_Pathway,
		Dimension_SideEffect,Dimension_Phenotype,Dimension_Gene, Dimension_Disease)
	Dimension_Range2=min(Dimension_MolecularStructure,Dimension_Target,Dimension_Pathway,
		Dimension_SideEffect,Dimension_Phenotype,Dimension_Gene, Dimension_Disease)

	Betas_MolecularStructure=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_Target=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_Pathway=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_SideEffect=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_Phenotype=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_Gene=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_Disease=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]

	Alphas_MolecularStructure=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_Target=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_Pathway=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_SideEffect=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_Phenotype=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_Gene=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_Disease=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]

	Gammas=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	
	
	Betas_MolecularStructure=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_Target=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_Pathway=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_SideEffect=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_Phenotype=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_Gene=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Betas_Disease=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]

	Alphas_MolecularStructure=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_Target=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_Pathway=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_SideEffect=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_Phenotype=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_Gene=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	Alphas_Disease=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]

	Gammas=[0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,10000]
	# 一组一组讨论

	Common_Dimention1=list(map(int, [(i*Dimension_Range1/60) for i in range(1,60)]))
	Common_Dimention1=np.array((npp.arange(100,2001,100)))
	# print(Common_Dimention1)
	# Common_Dimention2=list(map(int, [(i*Dimension_Range2/20) for i in range(1,21)]))
	# print(Dimension_Range1)
	# print(Common_Dimention1)
	# print(Dimension_Range2)
	# print(Common_Dimention2)
	

	# mainly discuss the parameters of Target on the results
	Beta_MolecularStructure=0.05;Alpha_MolecularStructure=0.001 ########################
	Beta_Target=0.5;Alpha_Target=5
	Beta_Pathway=0.05;Alpha_Pathway=0.001
	Beta_SideEffect=0.05;Alpha_SideEffect=0.0001
	Beta_Phenotype=5;Alpha_Phenotype=0.1
	Beta_Gene=0.01;Alpha_Gene=0.005
	Beta_Disease=1;Alpha_Disease=100
	Gammas=[0.5]
	Common_Dimention1=[900]

	for Common_Dimention in Common_Dimention1:
		for gamma in Gammas:
			# Initialize drug attribute matrices (totally seven attribute by DrugAttributeExistingMatrix)
			X_MolecularStructure=np.random.rand(DrugNum,Dimension_MolecularStructure)
			X_MolecularStructureMissingDrugRepresentation=np.dot(H_MissingIndicatingMolecularStructureMatrix,X_MolecularStructure)
			X_MolecularStructure=np.dot(ExistingRecoveryIndicatingMolecularStructureMatrix,DrugMolecularStructureExistingMatrix)+\
				np.dot(MissingRecoveryIndicatingMolecularStructureMatrix,X_MolecularStructureMissingDrugRepresentation)
			# print(X_MolecularStructure)

			X_Target=np.random.rand(DrugNum,Dimension_Target)
			X_TargetMissingDrugRepresentation=np.dot(H_MissingIndicatingTargetMatrix,X_Target)
			X_Target=np.dot(ExistingRecoveryIndicatingTargetMatrix,DrugTargetExistingMatrix)+\
				np.dot(MissingRecoveryIndicatingTargetMatrix,X_TargetMissingDrugRepresentation)

			X_Pathway=np.random.rand(DrugNum,Dimension_Pathway)
			X_PathwayMissingDrugRepresentation=np.dot(H_MissingIndicatingPathwayMatrix,X_Pathway)
			X_Pathway=np.dot(ExistingRecoveryIndicatingPathwayMatrix,DrugPathwayExistingMatrix)+\
				np.dot(MissingRecoveryIndicatingPathwayMatrix,X_PathwayMissingDrugRepresentation)


			X_SideEffect=(np.random.rand(DrugNum,Dimension_SideEffect))
			X_SideEffectMissingDrugRepresentation=np.dot(H_MissingIndicatingSideEffectMatrix,X_SideEffect)
			X_SideEffect=np.dot(ExistingRecoveryIndicatingSideEffectMatrix,DrugSideEffectExistingMatrix)+\
				np.dot(MissingRecoveryIndicatingSideEffectMatrix,X_SideEffectMissingDrugRepresentation)


			X_Phenotype=(np.random.rand(DrugNum,Dimension_Phenotype))
			X_PhenotypeMissingDrugRepresentation=np.dot(H_MissingIndicatingPhenotypeMatrix,X_Phenotype)
			X_Phenotype=np.dot(ExistingRecoveryIndicatingPhenotypeMatrix,DrugPhenotypeExistingMatrix)+\
				np.dot(MissingRecoveryIndicatingPhenotypeMatrix,X_PhenotypeMissingDrugRepresentation)
			

			X_Gene=(np.random.rand(DrugNum,Dimension_Gene))
			X_GeneMissingDrugRepresentation=np.dot(H_MissingIndicatingGeneMatrix,X_Gene)
			X_Gene=np.dot(ExistingRecoveryIndicatingGeneMatrix,DrugGeneExistingMatrix)+\
				np.dot(MissingRecoveryIndicatingGeneMatrix,X_GeneMissingDrugRepresentation)


			X_Disease=(np.random.rand(DrugNum,Dimension_Disease))
			X_DiseaseMissingDrugRepresentation=np.dot(H_MissingIndicatingDiseaseMatrix,X_Disease)
			X_Disease=np.dot(ExistingRecoveryIndicatingDiseaseMatrix,DrugDiseaseExistingMatrix)+\
				np.dot(MissingRecoveryIndicatingDiseaseMatrix,X_DiseaseMissingDrugRepresentation)
			# print(X_Target)
			# Initialize commom attribute matrix P
			P_CommonAttributeMatrix=(np.random.rand(DrugNum,Common_Dimention))
			# Initialize specific attribute matrix Q
			Q_MolecularStructure=(np.random.rand(DrugNum,Common_Dimention))
			Q_Target=(np.random.rand(DrugNum,Common_Dimention))
			Q_Pathway=(np.random.rand(DrugNum,Common_Dimention))
			Q_SideEffect=(np.random.rand(DrugNum,Common_Dimention))
			Q_Phenotype=(np.random.rand(DrugNum,Common_Dimention))
			Q_Gene=(np.random.rand(DrugNum,Common_Dimention))
			Q_Disease=(np.random.rand(DrugNum,Common_Dimention))
			
			# Initialize matrix U
			U_MolecularStructure=ReturnDiagonalMatrixofNoSquareMatrix(Common_Dimention,Dimension_MolecularStructure)
			U_Target=ReturnDiagonalMatrixofNoSquareMatrix(Common_Dimention,Dimension_Target)
			U_Pathway=ReturnDiagonalMatrixofNoSquareMatrix(Common_Dimention,Dimension_Pathway)
			U_SideEffect=ReturnDiagonalMatrixofNoSquareMatrix(Common_Dimention,Dimension_SideEffect)
			U_Phenotype=ReturnDiagonalMatrixofNoSquareMatrix(Common_Dimention,Dimension_Phenotype)
			U_Gene=ReturnDiagonalMatrixofNoSquareMatrix(Common_Dimention,Dimension_Gene)
			U_Disease=ReturnDiagonalMatrixofNoSquareMatrix(Common_Dimention,Dimension_Disease)

			# Initialize matrix Hat_U
			Hat_U_MolecularStructure=U_MolecularStructure
			Hat_U_Target=U_Target
			Hat_U_Pathway=U_Pathway
			Hat_U_SideEffect=U_SideEffect
			Hat_U_Phenotype=U_Phenotype
			Hat_U_Gene=U_Gene
			Hat_U_Disease=U_Disease


			# Initialize matrix Lambda
			Lambda_MolecularStructure=np.zeros((Common_Dimention,Dimension_MolecularStructure))
			Lambda_Target=np.zeros((Common_Dimention,Dimension_Target))
			Lambda_Pathway=np.zeros((Common_Dimention,Dimension_Pathway))
			Lambda_SideEffect=(np.zeros((Common_Dimention,Dimension_SideEffect)))
			Lambda_Phenotype=(np.zeros((Common_Dimention,Dimension_Phenotype)))
			Lambda_Gene=np.zeros((Common_Dimention,Dimension_Gene))
			Lambda_Disease=np.zeros((Common_Dimention,Dimension_Disease))


			Rou=0.00001
			Max_Rou=1000000
			tau=1.01
			MaxIter=20

			Loss_Last=1000000
			Loss_Threthold=0.0001
			for Iter in range(MaxIter):
				# Update X_Attribute
				X_MolecularStructure=UpdateX_AttributeFunction(X_MolecularStructure,
					P_CommonAttributeMatrix,Q_MolecularStructure,U_MolecularStructure,Beta_MolecularStructure)
				X_Target=UpdateX_AttributeFunction(X_Target,
					P_CommonAttributeMatrix,Q_Target,U_Target,Beta_Target)
				X_Pathway=UpdateX_AttributeFunction(X_Pathway,
					P_CommonAttributeMatrix,Q_Pathway,U_Pathway,Beta_Pathway)		
				X_SideEffect=UpdateX_AttributeFunction(X_SideEffect,
					P_CommonAttributeMatrix,Q_SideEffect,U_SideEffect,Beta_SideEffect)
				X_Phenotype=UpdateX_AttributeFunction(X_Phenotype,
					P_CommonAttributeMatrix,Q_Phenotype,U_Phenotype,Beta_Phenotype)
				X_Gene=UpdateX_AttributeFunction(X_Gene,
					P_CommonAttributeMatrix,Q_Gene,U_Gene,Beta_Gene)						
				X_Disease=UpdateX_AttributeFunction(X_Disease,
					P_CommonAttributeMatrix,Q_Disease,U_Disease,Beta_Disease)		
				# Revover X_Attribute
				X_MolecularStructureMissingDrugRepresentation=np.dot(H_MissingIndicatingMolecularStructureMatrix,X_MolecularStructure)
				X_MolecularStructure=np.dot(ExistingRecoveryIndicatingMolecularStructureMatrix,DrugMolecularStructureExistingMatrix)+\
					np.dot(MissingRecoveryIndicatingMolecularStructureMatrix,X_MolecularStructureMissingDrugRepresentation)
					
				X_TargetMissingDrugRepresentation=np.dot(H_MissingIndicatingTargetMatrix,X_Target)
				X_Target=np.dot(ExistingRecoveryIndicatingTargetMatrix,DrugTargetExistingMatrix)+\
					np.dot(MissingRecoveryIndicatingTargetMatrix,X_TargetMissingDrugRepresentation)

				X_PathwayMissingDrugRepresentation=np.dot(H_MissingIndicatingPathwayMatrix,X_Pathway)
				X_Pathway=np.dot(ExistingRecoveryIndicatingPathwayMatrix,DrugPathwayExistingMatrix)+\
					np.dot(MissingRecoveryIndicatingPathwayMatrix,X_PathwayMissingDrugRepresentation)

				X_SideEffectMissingDrugRepresentation=np.dot(H_MissingIndicatingSideEffectMatrix,X_SideEffect)
				X_SideEffect=np.dot(ExistingRecoveryIndicatingSideEffectMatrix,DrugSideEffectExistingMatrix)+\
					np.dot(MissingRecoveryIndicatingSideEffectMatrix,X_SideEffectMissingDrugRepresentation)

				X_PhenotypeMissingDrugRepresentation=np.dot(H_MissingIndicatingPhenotypeMatrix,X_Phenotype)
				X_Phenotype=np.dot(ExistingRecoveryIndicatingPhenotypeMatrix,DrugPhenotypeExistingMatrix)+\
					np.dot(MissingRecoveryIndicatingPhenotypeMatrix,X_PhenotypeMissingDrugRepresentation)
	
				X_GeneMissingDrugRepresentation=np.dot(H_MissingIndicatingGeneMatrix,X_Gene)
				X_Gene=np.dot(ExistingRecoveryIndicatingGeneMatrix,DrugGeneExistingMatrix)+\
					np.dot(MissingRecoveryIndicatingGeneMatrix,X_GeneMissingDrugRepresentation)

				X_DiseaseMissingDrugRepresentation=np.dot(H_MissingIndicatingDiseaseMatrix,X_Disease)
				X_Disease=np.dot(ExistingRecoveryIndicatingDiseaseMatrix,DrugDiseaseExistingMatrix)+\
					np.dot(MissingRecoveryIndicatingDiseaseMatrix,X_DiseaseMissingDrugRepresentation)
				
				# print('Iter=%d X_Attribute have been updated' %(Iter))

				# Update Common Attribute Representation
				F_MolecularStructure=ReturnStandardConsinofX_AttributeMatrix(X_MolecularStructure)
				F_Target=ReturnStandardConsinofX_AttributeMatrix(X_Target)
				F_Pathway=ReturnStandardConsinofX_AttributeMatrix(X_Pathway)	
				F_SideEffect=ReturnStandardConsinofX_AttributeMatrix(X_SideEffect)
				F_Phenotype=ReturnStandardConsinofX_AttributeMatrix(X_Phenotype)
				F_Gene=ReturnStandardConsinofX_AttributeMatrix(X_Gene)								
				F_Disease=ReturnStandardConsinofX_AttributeMatrix(X_Disease)	

				DiagonalF_MolecularStructure=CalculateDiagonalMatrixofSumofRow(F_MolecularStructure)
				DiagonalF_Target=CalculateDiagonalMatrixofSumofRow(F_Target)
				DiagonalF_Pathway=CalculateDiagonalMatrixofSumofRow(F_Pathway)
				DiagonalF_SideEffect=CalculateDiagonalMatrixofSumofRow(F_SideEffect)
				DiagonalF_Phenotype=CalculateDiagonalMatrixofSumofRow(F_Phenotype)
				DiagonalF_Gene=CalculateDiagonalMatrixofSumofRow(F_Gene)
				DiagonalF_Disease=CalculateDiagonalMatrixofSumofRow(F_Disease)

				Delta_MolecularStructure=DiagonalF_MolecularStructure-F_MolecularStructure
				Delta_Target=DiagonalF_Target-F_Target
				Delta_Pathway=DiagonalF_Pathway-F_Pathway
				Delta_SideEffect=DiagonalF_SideEffect-F_SideEffect
				Delta_Phenotype=DiagonalF_Phenotype-F_Phenotype
				Delta_Gene=DiagonalF_Gene-F_Gene
				Delta_Disease=DiagonalF_Disease-F_Disease

				ListBeta=[Beta_MolecularStructure,Beta_Target,Beta_Pathway,
					Beta_SideEffect,Beta_Phenotype,Beta_Gene,Beta_Disease]
				ListDeltaMatrix=[Delta_MolecularStructure,Delta_Target,Delta_Pathway,
					Delta_SideEffect,Delta_Phenotype,Delta_Gene,Delta_Disease]
				List_X_Attribute=[X_MolecularStructure,X_Target,X_Pathway,X_SideEffect,X_Phenotype,X_Gene,X_Disease]
				List_Q_Matrix=[Q_MolecularStructure,Q_Target,Q_Pathway,Q_SideEffect,Q_Phenotype,Q_Gene,Q_Disease]
				List_F_Attribute=[F_MolecularStructure,F_Target,F_Pathway,F_SideEffect,F_Phenotype,F_Gene,F_Disease]
				List_DiagonalF_Attribute=[DiagonalF_MolecularStructure,DiagonalF_Target,DiagonalF_Pathway,DiagonalF_SideEffect,\
					DiagonalF_Phenotype,DiagonalF_Gene,DiagonalF_Disease]
				List_U_Matrix=[U_MolecularStructure,U_Target,U_Pathway,U_SideEffect,U_Phenotype,U_Gene,U_Disease]
				
				NumeratorforUpdatingP,DenominatorforUpdatingP=CalculateNumeratorandDenominatorforUpdatingP(List_X_Attribute,
					P_CommonAttributeMatrix,List_Q_Matrix,List_U_Matrix,List_F_Attribute,List_DiagonalF_Attribute,ListBeta)
				
				P_CommonAttributeMatrix=P_CommonAttributeMatrix*NumeratorforUpdatingP/DenominatorforUpdatingP
				
				P_CommonAttributeMatrix=np.where(DenominatorforUpdatingP!=0,P_CommonAttributeMatrix,
					np.zeros_like(P_CommonAttributeMatrix))

				# print(P_CommonAttributeMatrix)
				# ALL_BetaDeltaMatrix=CalculateBetaAndDeltaMatrix(ListBeta,ListDeltaMatrix)
				# EigenVectorTheta1,DecompositionMatrixA=EigenDecompositionX_Attribute(ALL_BetaDeltaMatrix)
				# ALL_U_MatrixandItsTranspose=CalculateU_MatrixandTranspose(List_U_Matrix)
				# ALL_U_MatrixandItsTranspose=np.linalg.inv(ALL_U_MatrixandItsTranspose)
				# EigenVectorTheta2,DecompositionMatrixB=EigenDecompositionX_Attribute(ALL_U_MatrixandItsTranspose)
				# RightValueforUpdatingP=CalculateRightC_mForUpdateMatrixP(List_X_Attribute,
				# 	List_Q_Matrix,List_U_Matrix,ListDeltaMatrix,ListBeta)
				# RightValueforUpdatingP=np.dot(RightValueforUpdatingP,ALL_U_MatrixandItsTranspose)
							
				# Numerator=np.dot(np.dot(DecompositionMatrixA.transpose(),RightValueforUpdatingP),DecompositionMatrixB)
				# ALL_Ones=np.ones((DrugNum,Common_Dimention))
				# Theta1Theta2Matrix=np.dot(EigenVectorTheta1.transpose(),EigenVectorTheta2)
				# Y_Matrix=Numerator/(ALL_Ones+Theta1Theta2Matrix)

				# P_CommonAttributeMatrix=np.dot(np.dot(DecompositionMatrixA,Y_Matrix),DecompositionMatrixB.transpose())

				# print(P_CommonAttributeMatrix)
				# print('Iter=%d P have been updated' %(Iter))
				# Update matrix Q for each attribute

				Q_MolecularStructure=UpdateMatrixQ_Attribute(X_MolecularStructure,U_MolecularStructure,P_CommonAttributeMatrix,
					F_MolecularStructure,DiagonalF_MolecularStructure,Q_MolecularStructure,List_Q_Matrix,gamma,Beta_MolecularStructure)
				Q_Target=UpdateMatrixQ_Attribute(X_Target,U_Target,P_CommonAttributeMatrix,
					F_Target,DiagonalF_Target,Q_Target,List_Q_Matrix,gamma,Beta_Target)
				Q_Pathway=UpdateMatrixQ_Attribute(X_Pathway,U_Pathway,P_CommonAttributeMatrix,
					F_Pathway,DiagonalF_Pathway,Q_Pathway,List_Q_Matrix,gamma,Beta_Pathway)
				Q_SideEffect=UpdateMatrixQ_Attribute(X_SideEffect,U_SideEffect,P_CommonAttributeMatrix,
					F_SideEffect,DiagonalF_SideEffect,Q_SideEffect,List_Q_Matrix,gamma,Beta_SideEffect)
				Q_Phenotype=UpdateMatrixQ_Attribute(X_Phenotype,U_Phenotype,P_CommonAttributeMatrix,
					F_Phenotype,DiagonalF_Phenotype,Q_Phenotype,List_Q_Matrix,gamma,Beta_Phenotype)
				Q_Gene=UpdateMatrixQ_Attribute(X_Gene,U_Gene,P_CommonAttributeMatrix,
					F_Gene,DiagonalF_Gene,Q_Gene,List_Q_Matrix,gamma,Beta_Gene)
				Q_Disease=UpdateMatrixQ_Attribute(X_Disease,U_Disease,P_CommonAttributeMatrix,
					F_Disease,DiagonalF_Disease,Q_Disease,List_Q_Matrix,gamma,Beta_Disease)
				# print(Q_MolecularStructure)
				# print('Iter=%d Q have been updated' %(Iter))
				# Update matrix Hat_U for each attribute
				Hat_U_MolecularStructure=UpdateMatrixHatU_Attribute_Updation2(X_MolecularStructure,P_CommonAttributeMatrix,
					Q_MolecularStructure,Rou,Lambda_MolecularStructure,U_MolecularStructure,Hat_U_MolecularStructure)
				Hat_U_Target=UpdateMatrixHatU_Attribute_Updation2(X_Target,P_CommonAttributeMatrix,
					Q_Target,Rou,Lambda_Target,U_Target,Hat_U_Target)
				Hat_U_Pathway=UpdateMatrixHatU_Attribute_Updation2(X_Pathway,P_CommonAttributeMatrix,
					Q_Pathway,Rou,Lambda_Pathway,U_Pathway,Hat_U_Pathway)
				Hat_U_SideEffect=UpdateMatrixHatU_Attribute_Updation2(X_SideEffect,P_CommonAttributeMatrix,
					Q_SideEffect,Rou,Lambda_SideEffect,U_SideEffect,Hat_U_SideEffect)
				Hat_U_Phenotype=UpdateMatrixHatU_Attribute_Updation2(X_Phenotype,P_CommonAttributeMatrix,
					Q_Phenotype,Rou,Lambda_Phenotype,U_Phenotype,Hat_U_Phenotype)
				Hat_U_Gene=UpdateMatrixHatU_Attribute_Updation2(X_Gene,P_CommonAttributeMatrix,
					Q_Gene,Rou,Lambda_Gene,U_Gene,Hat_U_Gene)
				Hat_U_Disease=UpdateMatrixHatU_Attribute_Updation2(X_Disease,P_CommonAttributeMatrix,
					Q_Disease,Rou,Lambda_Disease,U_Disease,Hat_U_Disease)
				# print(Hat_U_MolecularStructure)
				# update Matrix U for each attribute
				U_MolecularStructure=UpdateMatrixU_Attribute(Alpha_MolecularStructure,
					Hat_U_MolecularStructure,Lambda_MolecularStructure,Rou)
				U_Target=UpdateMatrixU_Attribute(Alpha_Target,Hat_U_Target,Lambda_Target,Rou)
				U_Pathway=UpdateMatrixU_Attribute(Alpha_Pathway,Hat_U_Pathway,Lambda_Pathway,Rou)
				U_SideEffect=UpdateMatrixU_Attribute(Alpha_SideEffect,Hat_U_SideEffect,Lambda_SideEffect,Rou)
				U_Phenotype=UpdateMatrixU_Attribute(Alpha_Phenotype,Hat_U_Phenotype,Lambda_Phenotype,Rou)
				U_Gene=UpdateMatrixU_Attribute(Alpha_Gene,Hat_U_Gene,Lambda_Gene,Rou)
				U_Disease=UpdateMatrixU_Attribute(Alpha_Disease,Hat_U_Disease,Lambda_Disease,Rou)

				Lambda_MolecularStructure=Lambda_MolecularStructure+Rou*(U_MolecularStructure-Hat_U_MolecularStructure)
				Lambda_Target=Lambda_Target+Rou*(U_Target-Hat_U_Target)
				Lambda_Pathway=Lambda_Pathway+Rou*(U_Pathway-Hat_U_Pathway)
				Lambda_SideEffect=Lambda_SideEffect+Rou*(U_SideEffect-Hat_U_SideEffect)
				Lambda_Phenotype=Lambda_Phenotype+Rou*(U_Phenotype-Hat_U_Phenotype)
				Lambda_Gene=Lambda_Gene+Rou*(U_Gene-Hat_U_Gene)
				Lambda_Disease=Lambda_Disease+Rou*(U_Disease-Hat_U_Disease)
				# print('Iter=%d U have been updated' %(Iter))
				Rou=min(tau*Rou,Max_Rou)


				Loss_MolecularStructure=np.linalg.norm(X_MolecularStructure-np.dot(P_CommonAttributeMatrix+Q_MolecularStructure,U_MolecularStructure))
				Loss_Target=np.linalg.norm(X_Target-np.dot(P_CommonAttributeMatrix+Q_Target,U_Target))
				Loss_Pathway=np.linalg.norm(X_Pathway-np.dot(P_CommonAttributeMatrix+Q_Pathway,U_Pathway))
				Loss_SideEffect=np.linalg.norm(X_SideEffect-np.dot(P_CommonAttributeMatrix+Q_SideEffect,U_SideEffect))
				Loss_Phenotype=np.linalg.norm(X_Phenotype-np.dot(P_CommonAttributeMatrix+Q_Phenotype,U_Phenotype))
				Loss_Gene=np.linalg.norm(X_Gene-np.dot(P_CommonAttributeMatrix+Q_Gene,U_Gene))
				Loss_Disease=np.linalg.norm(X_Disease-np.dot(P_CommonAttributeMatrix+Q_Disease,U_Disease))

				LossTotal=Loss_MolecularStructure+Loss_Target+Loss_Pathway+Loss_SideEffect+Loss_Phenotype+Loss_Gene+Loss_Disease
				print('Iter=%d, Loss=%.4f' %(Iter,LossTotal))
				if abs(Loss_Last-LossTotal)>Loss_Threthold:
					Loss_Last=LossTotal
				elif abs(Loss_Last-LossTotal)<=Loss_Threthold:
					break



			print('Output Recovered Attribute Matrices of Drugs !!!')
			Address='./RecoveredMultiAttributeRepresentations/'
			if not os.path.exists(Address):
				os.mkdir(Address)
			np.save(Address+'X_MolecularStructureRecovered',X_MolecularStructure)
			np.savetxt(Address+'X_MolecularStructureRecovered.txt',X_MolecularStructure)
			
			np.save(Address+'X_TargetRecovered',X_Target)
			np.savetxt(Address+'X_TargetRecovered.txt',X_Target)

			np.save(Address+'X_PathwayRecovered',X_Pathway)
			np.savetxt(Address+'X_PathwayRecovered.txt',X_Pathway)

			np.save(Address+'X_SideEffectRecovered',X_SideEffect)
			np.savetxt(Address+'X_SideEffectRecovered.txt',X_SideEffect)

			np.save(Address+'X_PhenotypeRecovered',X_Phenotype)
			np.savetxt(Address+'X_PhenotypeRecovered.txt',X_Phenotype)
			
			np.save(Address+'X_GeneRecovered',X_Gene)
			np.savetxt(Address+'X_GeneRecovered.txt',X_Gene)

			np.save(Address+'X_DiseaseRecovered',X_Disease)
			np.savetxt(Address+'X_DiseaseRecovered.txt',X_Disease)
			
			print('Shared and Specific Attribute Representations of Drugs!!!')
			Address='./SharedSpecificAttributeRepresentations/'
			if not os.path.exists(Address):
				os.mkdir(Address)
			np.save(Address+'P_CommonAttributeMatrix',P_CommonAttributeMatrix)
			np.savetxt(Address+'P_CommonAttributeMatrix.txt',P_CommonAttributeMatrix)


			np.save(Address+'Q_MolecularStructure',Q_MolecularStructure)
			np.savetxt(Address+'Q_MolecularStructure.txt',Q_MolecularStructure)
			
			np.save(Address+'Q_Target',Q_Target)
			np.savetxt(Address+'Q_Target.txt',Q_Target)

			np.save(Address+'Q_Pathway',Q_Pathway)
			np.savetxt(Address+'Q_PathwayFilled.txt',Q_Pathway)

			np.save(Address+'Q_SideEffect',Q_SideEffect)
			np.savetxt(Address+'Q_SideEffect.txt',Q_SideEffect)

			np.save(Address+'Q_Phenotype',Q_Phenotype)
			np.savetxt(Address+'Q_Phenotype.txt',Q_Phenotype)
			
			np.save(Address+'Q_Gene',Q_Gene)
			np.savetxt(Address+'Q_Gene.txt',Q_Gene)

			np.save(Address+'Q_Disease',Q_Disease)
			np.savetxt(Address+'Q_Disease.txt',Q_Disease)
			



if __name__ == '__main__':


	DrugMolecularStructureMatrixAddress='./1. MolecularStructure/DrugMolecularStructureMatrix.txt'
	DrugMolecularStructureMatrix=LoadDrugAttributeMatrix(DrugMolecularStructureMatrixAddress)
	# print(np.shape(DrugMolecularStructureMatrix))
	MolecularStructureDrugIndex,NoMolecularStructureDrugIndex=GetExistingMissingDrugAttributeMatrixIndex(DrugMolecularStructureMatrix)
	DrugMolecularStructureExistingMatrix,H_ExistingIndicatingMolecularStructureMatrix,H_MissingIndicatingMolecularStructureMatrix=\
		GetExistingDrugAttributeMatrixwithIndicatingMatrix(DrugMolecularStructureMatrix,MolecularStructureDrugIndex,NoMolecularStructureDrugIndex)
	# print(H_MissingIndicatingMolecularStructureMatrix)
	ExistingRecoveryIndicatingMolecularStructureMatrix,MissingRecoveryIndicatingMolecularStructureMatrix=\
		GetExistingMissingRecoveryIndicatingMatrix(MolecularStructureDrugIndex,NoMolecularStructureDrugIndex)
	# print(ExistingRecoveryIndicatingMolecularStructureMatrix,MissingRecoveryIndicatingMolecularStructureMatrix)


	DrugTargetMatrixAddress='./2. Target/DrugTargetMatrix.txt'
	DrugTargetMatrix=LoadDrugAttributeMatrix(DrugTargetMatrixAddress)
	# print(np.shape(DrugTargetMatrix))
	TargetDrugIndex,NoTargetDrugIndex=GetExistingMissingDrugAttributeMatrixIndex(DrugTargetMatrix)
	DrugTargetExistingMatrix,H_ExistingIndicatingTargetMatrix,H_MissingIndicatingTargetMatrix=\
		GetExistingDrugAttributeMatrixwithIndicatingMatrix(DrugTargetMatrix,TargetDrugIndex,NoTargetDrugIndex)
	ExistingRecoveryIndicatingTargetMatrix,MissingRecoveryIndicatingTargetMatrix=\
		GetExistingMissingRecoveryIndicatingMatrix(TargetDrugIndex,NoTargetDrugIndex)
	# print(ExistingRecoveryIndicatingTargetMatrix*DrugTargetExistingMatrix)


	DrugPathwayMatrixAddress='./3. Pathway/DrugPathwayMatrix.txt'
	DrugPathwayMatrix=LoadDrugAttributeMatrix(DrugPathwayMatrixAddress)
	# print(np.shape(DrugPathwayMatrix))
	PathwayDrugIndex,NoPathwayDrugIndex=GetExistingMissingDrugAttributeMatrixIndex(DrugPathwayMatrix)
	DrugPathwayExistingMatrix,H_ExistingIndicatingPathwayMatrix,H_MissingIndicatingPathwayMatrix=\
		GetExistingDrugAttributeMatrixwithIndicatingMatrix(DrugPathwayMatrix,PathwayDrugIndex,NoPathwayDrugIndex)
	# print(np.shape(DrugPathwayExistingMatrix),np.shape(H_ExistingIndicatingPathwayMatrix))
	ExistingRecoveryIndicatingPathwayMatrix,MissingRecoveryIndicatingPathwayMatrix=\
		GetExistingMissingRecoveryIndicatingMatrix(PathwayDrugIndex,NoPathwayDrugIndex)


	DrugSideEffectMatrixAddress='./4. SideEffect/DrugSideEffectMatrix.txt'
	DrugSideEffectMatrix=LoadDrugAttributeMatrix(DrugSideEffectMatrixAddress)	
	# print(np.shape(DrugSideEffectMatrix))
	SideEffectDrugIndex,NoSideEffectDrugIndex=GetExistingMissingDrugAttributeMatrixIndex(DrugSideEffectMatrix)
	#print(NoSideEffectDrugIndex)
	DrugSideEffectExistingMatrix,H_ExistingIndicatingSideEffectMatrix,H_MissingIndicatingSideEffectMatrix=\
		GetExistingDrugAttributeMatrixwithIndicatingMatrix(DrugSideEffectMatrix,SideEffectDrugIndex,NoSideEffectDrugIndex)
	ExistingRecoveryIndicatingSideEffectMatrix,MissingRecoveryIndicatingSideEffectMatrix=\
		GetExistingMissingRecoveryIndicatingMatrix(SideEffectDrugIndex,NoSideEffectDrugIndex)


	DrugPhenotypeMatrixAddress='./5. Phenotype/DrugPhenotypeMatrix.txt'
	DrugPhenotypeMatrix=LoadDrugAttributeMatrix(DrugPhenotypeMatrixAddress)
	# print(np.shape(DrugPhenotypeMatrix))
	PhenotypeDrugIndex,NoPhenotypeDrugIndex=GetExistingMissingDrugAttributeMatrixIndex(DrugPhenotypeMatrix)
	DrugPhenotypeExistingMatrix,H_ExistingIndicatingPhenotypeMatrix,H_MissingIndicatingPhenotypeMatrix=\
		GetExistingDrugAttributeMatrixwithIndicatingMatrix(DrugPhenotypeMatrix,PhenotypeDrugIndex,NoPhenotypeDrugIndex)
	ExistingRecoveryIndicatingPhenotypeMatrix,MissingRecoveryIndicatingPhenotypeMatrix=\
		GetExistingMissingRecoveryIndicatingMatrix(PhenotypeDrugIndex,NoPhenotypeDrugIndex)


	DrugGeneMatrixAddress='./6. Gene/DrugGeneMatrix.txt'
	DrugGeneMatrix=LoadDrugAttributeMatrix(DrugGeneMatrixAddress)
	# print(np.shape(DrugGeneMatrix))
	GeneDrugIndex,NoGeneDrugIndex=GetExistingMissingDrugAttributeMatrixIndex(DrugGeneMatrix)
	DrugGeneExistingMatrix,H_ExistingIndicatingGeneMatrix,H_MissingIndicatingGeneMatrix=\
		GetExistingDrugAttributeMatrixwithIndicatingMatrix(DrugGeneMatrix,GeneDrugIndex,NoGeneDrugIndex)
	ExistingRecoveryIndicatingGeneMatrix,MissingRecoveryIndicatingGeneMatrix=\
		GetExistingMissingRecoveryIndicatingMatrix(GeneDrugIndex,NoGeneDrugIndex)


	DrugDiseaseMatrixAddress='./7. Disease/DrugDiseaseMatrix.txt'
	DrugDiseaseMatrix=LoadDrugAttributeMatrix(DrugDiseaseMatrixAddress)
	# print(np.shape(DrugDiseaseMatrix))
	DiseaseDrugIndex,NoDiseaseDrugIndex=GetExistingMissingDrugAttributeMatrixIndex(DrugDiseaseMatrix)
	DrugDiseaseExistingMatrix,H_ExistingIndicatinDiseaseMatrix,H_MissingIndicatingDiseaseMatrix=\
		GetExistingDrugAttributeMatrixwithIndicatingMatrix(DrugDiseaseMatrix,DiseaseDrugIndex,NoDiseaseDrugIndex)
	ExistingRecoveryIndicatingDiseaseMatrix,MissingRecoveryIndicatingDiseaseMatrix=\
		GetExistingMissingRecoveryIndicatingMatrix(DiseaseDrugIndex,NoDiseaseDrugIndex)	

	LearningProcessforUpdateParameters(DrugMolecularStructureExistingMatrix,H_ExistingIndicatingMolecularStructureMatrix,
		H_MissingIndicatingMolecularStructureMatrix,ExistingRecoveryIndicatingMolecularStructureMatrix,MissingRecoveryIndicatingMolecularStructureMatrix,
		DrugTargetExistingMatrix,H_ExistingIndicatingTargetMatrix,H_MissingIndicatingTargetMatrix,
		ExistingRecoveryIndicatingTargetMatrix,MissingRecoveryIndicatingTargetMatrix,
		DrugPathwayExistingMatrix,H_ExistingIndicatingPathwayMatrix,H_MissingIndicatingPathwayMatrix,
		ExistingRecoveryIndicatingPathwayMatrix,MissingRecoveryIndicatingPathwayMatrix,
		DrugSideEffectExistingMatrix,H_ExistingIndicatingSideEffectMatrix,H_MissingIndicatingSideEffectMatrix,
		ExistingRecoveryIndicatingSideEffectMatrix,MissingRecoveryIndicatingSideEffectMatrix,
		DrugPhenotypeExistingMatrix,H_ExistingIndicatingPhenotypeMatrix,H_MissingIndicatingPhenotypeMatrix,
		ExistingRecoveryIndicatingPhenotypeMatrix,MissingRecoveryIndicatingPhenotypeMatrix,
		DrugGeneExistingMatrix,H_ExistingIndicatingGeneMatrix,H_MissingIndicatingGeneMatrix,
		ExistingRecoveryIndicatingGeneMatrix,MissingRecoveryIndicatingGeneMatrix,
		DrugDiseaseExistingMatrix,H_ExistingIndicatinDiseaseMatrix,H_MissingIndicatingDiseaseMatrix,
		ExistingRecoveryIndicatingDiseaseMatrix,MissingRecoveryIndicatingDiseaseMatrix)


