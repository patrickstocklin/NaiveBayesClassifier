import os
import timeit
import sys
from collections import defaultdict
from collections import OrderedDict
import fileinput
import math
import numpy as np
from decimal import *

'''
RELATION FORMAT

@relation adult

@attribute age numeric
@attribute workclass { Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked }
@attribute fnlwgt numeric
@attribute education { Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool }
@attribute education-num numeric
@attribute marital-status { Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse }
@attribute occupation { Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces }
@attribute relationship { Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried}
@attribute race { White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black }
@attribute sex { Female, Male }
@attribute capital-gain numeric
@attribute capital-loss numeric
@attribute hours-per-week numeric
@attribute native-country { United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands }
@attribute class { >50K, <=50K }

Considering categorizing hours-per-week into part-time <35, full-time 35-44, over-time >45

'''

###########################################################################
'''
TO DO LIST:
	[X]	MAKE SURE ENTROPY-BASED CATEGORIZATION WORKS FOR: fnlwtg, education_num, capital_gain, capital_loss, hours_per_week
	[]	10-Fold Cross Validation: Write Scores and Calculate 
	[]	Write up
	[]	Modularize This Jank, but do this last
'''
###########################################################################

'''
Class for Adult Objects (stores all relevant info about adults)
'''
class Adult(object):
	def __init__(self, age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, adultClass):
		self.age = age
		self.workclass = workclass
		self.fnlwgt = fnlwgt
		self.education = education
		self.education_num = education_num
		self.marital_status = marital_status
		self.occupation = occupation
		self.relationship = relationship
		self.race = race
		self.sex = sex
		self.capital_gain = capital_gain
		self.capital_loss = capital_loss
		self.hours_per_week = hours_per_week
		self.native_country = native_country
		self.adultClass = adultClass
		#Determines if an adult object needs to be modified
		self.missingAttr = self.isMissingValue()

	def printAttributes(self):
		print("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s" % (self.age, self.workclass, self.fnlwgt, 
			self.education, self.education_num, self.marital_status, self.occupation, self.race, self.sex, self.capital_gain, 
			self.capital_loss, self.hours_per_week, self.native_country, self.adultClass))
	'''
	Method to determine if an adult object is missing any attributes
	'''
	def isMissingValue(self):
		if self.age == '?' or \
			self.workclass == '?' or \
			self.fnlwgt == '?' or \
			self.education == '?' or \
			self.education_num == '?' or \
			self.marital_status == '?' or \
			self.occupation == '?' or \
			self.relationship == '?' or \
			self.race == '?' or \
			self.sex == '?' or \
			self.capital_gain == '?' or \
			self.capital_loss == '?' or \
			self.hours_per_week == '?' or \
			self.native_country == '?':
				return True
		else:
			return False
	
'''
Model Object for methods
'''
class Model(object):
	
	#Adults Array that stores all Adult Objects
	Adults = []
	#Adults List to Train
	AdultsTrain = []
	#Adults List to Test
	AdultsTest = []
	#48842 Relations in Test Data
	noRelations = 0
	#How many Relations are Missing an Attribute
	noRelationsMissing = 0
	#How many Attributes are Missing
	noMissing = 0

	###########################################################################
	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	OUR MODEL (Contains extra stuff for Preprocessing, Core Information is Kept in __INIT__)
	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	#NUMERICAL
	age1 = defaultdict(int)
	age1Disc = {}
	age1avg = 0
	age2 = defaultdict(int)
	age2Disc = {}
	age2avg = 0
	#Bins we discretize our ages by
	ageCategories = []
	#This will be the array of 2-val tuples (Age, Class) for which we use to discretize our data
	ages = []
	
	#CATEGORICAL
	workclass1 = {'?':0, 'Private':0, 'Self-emp-not-inc':0, 
			'Self-emp-inc':0, 'Federal-gov':0, 'Local-gov':0,
			'State-gov':0, 'Without-pay':0, 'Never-worked':0}
	workclass1mode = ''
	workclass2 = {'?':0, 'Private':0, 'Self-emp-not-inc':0, 
			'Self-emp-inc':0, 'Federal-gov':0, 'Local-gov':0,
			'State-gov':0, 'Without-pay':0, 'Never-worked':0}
	workclass2mode = ''

	#NUMERICAL
	fnlwgt1 = defaultdict(int)
	fnlwgt1Disc = {}
	fnlwgt1avg = 0
	fnlwgt2 = defaultdict(int)
	fnlwgt2Disc = {}
	fnlwgt2avg = 0
	#Bins we discretize our fnlwgts by
	fnlwgtCategories = []
	#This will be the array of 2-val tuples (Fnlwgt, Class) for which we use to discretize our data
	fnlwgts = []

	#CATEGORICAL
	education1 = {'?':0, 'Bachelors':0, 'Some-college':0, '11th':0, 
			'HS-grad':0, 'Prof-school':0, 'Assoc-acdm':0, 'Assoc-voc':0, 
			'9th':0, '7th-8th':0, '12th':0, 'Masters':0, '1st-4th':0, 
			'10th':0, 'Doctorate':0, '5th-6th':0, 'Preschool':0}
	education1mode = ''
	education2 = {'?':0, 'Bachelors':0, 'Some-college':0, '11th':0, 
			'HS-grad':0, 'Prof-school':0, 'Assoc-acdm':0, 'Assoc-voc':0, 
			'9th':0, '7th-8th':0, '12th':0, 'Masters':0, '1st-4th':0, 
			'10th':0, 'Doctorate':0, '5th-6th':0, 'Preschool':0}
	education2mode = ''

	#NUMERICAL
	education_num1 = defaultdict(int)
	education_num1Disc = {}
	education_num1avg = 0
	education_num2 = defaultdict(int)
	education_num2Disc = {}
	education_num2avg = 0
	#Bins we discretize our education_nums by
	education_numCategories = []
	#This will be the array of 2-val tuples (Fnlwgt, Class) for which we use to discretize our data
	education_nums = []

	#CATEGORICAL
	marital_status1 = {'?':0, 'Married-civ-spouse':0, 'Divorced':0, 
			'Never-married':0, 'Separated':0, 'Widowed':0, 
			'Married-spouse-absent':0, 'Married-AF-spouse':0}
	marital_status1mode = ''
	marital_status2 = {'?':0, 'Married-civ-spouse':0, 'Divorced':0, 
			'Never-married':0, 'Separated':0, 'Widowed':0, 
			'Married-spouse-absent':0, 'Married-AF-spouse':0}
	marital_status2mode = ''


	#CATEGORICAL
	occupation1 = {'?':0, 'Tech-support':0, 'Craft-repair':0, 
			'Other-service':0, 'Sales':0, 'Exec-managerial':0, 
			'Prof-specialty':0, 'Handlers-cleaners':0, 'Machine-op-inspct':0, 
			'Adm-clerical':0, 'Farming-fishing':0, 'Transport-moving':0, 
			'Priv-house-serv':0, 'Protective-serv':0, 'Armed-Forces':0}
	occupation1mode = ''
	occupation2 = {'?':0, 'Tech-support':0, 'Craft-repair':0, 
			'Other-service':0, 'Sales':0, 'Exec-managerial':0, 
			'Prof-specialty':0, 'Handlers-cleaners':0, 'Machine-op-inspct':0, 
			'Adm-clerical':0, 'Farming-fishing':0, 'Transport-moving':0, 
			'Priv-house-serv':0, 'Protective-serv':0, 'Armed-Forces':0}
	occupation2mode = ''

	#CATEGORICAL
	relationship1 = {'?':0, 'Wife':0, 'Own-child':0, 'Husband':0, 
			'Not-in-family':0, 'Other-relative':0, 'Unmarried':0}
	relationship1mode = ''
	relationship2 = {'?':0, 'Wife':0, 'Own-child':0, 'Husband':0, 
			'Not-in-family':0, 'Other-relative':0, 'Unmarried':0}
	relationship2mode = ''

	#CATEGORICAL
	race1 = {'?':0, 'White':0, 'Asian-Pac-Islander':0, 
			'Amer-Indian-Eskimo':0, 'Other':0, 'Black':0}
	race1mode = ''
	race2 = {'?':0, 'White':0, 'Asian-Pac-Islander':0, 
			'Amer-Indian-Eskimo':0, 'Other':0, 'Black':0}
	race2mode = ''

	#CATEGORICAL
	sex1 = {'?':0, 'Male':0, 'Female':0}
	sex1mode = ''
	sex2 = {'?':0, 'Male':0, 'Female':0}
	sex2mode = ''

	#NUMERICAL
	capital_gain1 = defaultdict(int)
	capital_gain1Disc = {}
	capital_gain1avg = 0
	capital_gain2 = defaultdict(int)
	capital_gain2Disc = {}
	capital_gain2avg = 0
	#Bins we discretize our capital_gains by
	capital_gainCategories = []
	#This will be the list of 2-value tuples (capgain, class) for which we use to discretize our data
	capital_gains = []

	#NUMERICAL
	capital_loss1 = defaultdict(int)
	capital_loss1Disc = {}
	capital_loss1avg = 0
	capital_loss2Disc = {}
	capital_loss2 = defaultdict(int)
	capital_loss2avg = 0
	#Bins we discretize our capital_losses by
	capital_lossCategories = []
	#This will be the list of 2-value tuples (caploss, class) for which we use to discretize our data
	capital_losses = []

	#NUMERICAL
	hours_per_week1 = defaultdict(int)
	hours_per_week1Disc = {}
	hours_per_week1avg = 0
	hours_per_week2 = defaultdict(int)
	hours_per_week2Disc = {}
	hours_per_week2avg = 0 
	#Bins we discretize our HPW into
	hours_per_weekCategories = []
	#This will be the list of 2-value tuples (hpw, class) for which we use to discretize our data
	hours_per_weeks = []

	#CATEGORICAL
	native_country1 = {'?':0, 'United-States':0, 'Cambodia':0, 'England':0, 
			'Puerto-Rico':0, 'Canada':0, 'Germany':0, 
			'Outlying-US(Guam-USVI-etc)':0, 'India':0, 'Japan':0, 
			'Greece':0, 'South':0, 'China':0, 'Cuba':0, 'Iran':0, 
			'Honduras':0, 'Philippines':0, 'Italy':0, 'Poland':0, 
			'Jamaica':0, 'Vietnam':0, 'Mexico':0, 'Portugal':0, 
			'Ireland':0, 'France':0, 'Dominican-Republic':0, 
			'Laos':0, 'Ecuador':0, 'Taiwan':0, 'Haiti':0, 
			'Columbia':0, 'Hungary':0, 'Guatemala':0, 'Nicaragua':0,
			'Scotland':0, 'Thailand':0, 'Yugoslavia':0, 'El-Salvador':0, 
			'Trinadad&Tobago':0, 'Peru':0, 'Hong':0, 'Holand-Netherlands':0}
	native_country1mode = ''
	native_country2 ={'?':0, 'United-States':0, 'Cambodia':0, 'England':0, 
			'Puerto-Rico':0, 'Canada':0, 'Germany':0, 
			'Outlying-US(Guam-USVI-etc)':0, 'India':0, 'Japan':0, 
			'Greece':0, 'South':0, 'China':0, 'Cuba':0, 'Iran':0, 
			'Honduras':0, 'Philippines':0, 'Italy':0, 'Poland':0, 
			'Jamaica':0, 'Vietnam':0, 'Mexico':0, 'Portugal':0, 
			'Ireland':0, 'France':0, 'Dominican-Republic':0, 
			'Laos':0, 'Ecuador':0, 'Taiwan':0, 'Haiti':0, 
			'Columbia':0, 'Hungary':0, 'Guatemala':0, 'Nicaragua':0,
			'Scotland':0, 'Thailand':0, 'Yugoslavia':0, 'El-Salvador':0, 
			'Trinadad&Tobago':0, 'Peru':0, 'Hong':0, 'Holand-Netherlands':0}
	native_country2mode = ''

	#Counters for Class 1: >50K, Class 2: <=50K 
	numOfClass1 = 0
	numOfClass2 = 0
	##########################################################################
	'''
	Constructor
	'''
	def __init__(self, num):
		self.name = "Naive Bayes Model " + num
		#Adults Array that stores all Adult Objects
		self.Adults = []
		#Adults List to Train
		self.AdultsTrain = []
		#Adults List to Test
		self.AdultsTest = []
		#How many relations there are
		self.noRelations = 0

		#Dictionaries:
		self.age1Disc = {}
		self.age2Disc = {}
		self.ageCategories = []

		self.workclass1 = {'?':0, 'Private':0, 'Self-emp-not-inc':0, 
			'Self-emp-inc':0, 'Federal-gov':0, 'Local-gov':0,
			'State-gov':0, 'Without-pay':0, 'Never-worked':0}
		self.workclass2 = {'?':0, 'Private':0, 'Self-emp-not-inc':0, 
			'Self-emp-inc':0, 'Federal-gov':0, 'Local-gov':0,
			'State-gov':0, 'Without-pay':0, 'Never-worked':0}

		self.fnlwgt1Disc = {}
		self.fnlwgt2Disc = {}
		self.fnlwgtCategories = []

		self.education1 = {'?':0, 'Bachelors':0, 'Some-college':0, '11th':0, 
			'HS-grad':0, 'Prof-school':0, 'Assoc-acdm':0, 'Assoc-voc':0, 
			'9th':0, '7th-8th':0, '12th':0, 'Masters':0, '1st-4th':0, 
			'10th':0, 'Doctorate':0, '5th-6th':0, 'Preschool':0}
		self.education2 = {'?':0, 'Bachelors':0, 'Some-college':0, '11th':0, 
			'HS-grad':0, 'Prof-school':0, 'Assoc-acdm':0, 'Assoc-voc':0, 
			'9th':0, '7th-8th':0, '12th':0, 'Masters':0, '1st-4th':0, 
			'10th':0, 'Doctorate':0, '5th-6th':0, 'Preschool':0}
	
		self.education_num1Disc = {}
		self.education_num2Disc = {}
		self.education_numCategories = []
	
		self.marital_status1 = {'?':0, 'Married-civ-spouse':0, 'Divorced':0, 
			'Never-married':0, 'Separated':0, 'Widowed':0, 
			'Married-spouse-absent':0, 'Married-AF-spouse':0}
		self.marital_status2 = {'?':0, 'Married-civ-spouse':0, 'Divorced':0, 
			'Never-married':0, 'Separated':0, 'Widowed':0, 
			'Married-spouse-absent':0, 'Married-AF-spouse':0}
	
		self.occupation1 = {'?':0, 'Tech-support':0, 'Craft-repair':0, 
			'Other-service':0, 'Sales':0, 'Exec-managerial':0, 
			'Prof-specialty':0, 'Handlers-cleaners':0, 'Machine-op-inspct':0, 
			'Adm-clerical':0, 'Farming-fishing':0, 'Transport-moving':0, 
			'Priv-house-serv':0, 'Protective-serv':0, 'Armed-Forces':0}
		self.occupation2 = {'?':0, 'Tech-support':0, 'Craft-repair':0, 
			'Other-service':0, 'Sales':0, 'Exec-managerial':0, 
			'Prof-specialty':0, 'Handlers-cleaners':0, 'Machine-op-inspct':0, 
			'Adm-clerical':0, 'Farming-fishing':0, 'Transport-moving':0, 
			'Priv-house-serv':0, 'Protective-serv':0, 'Armed-Forces':0}
	
		self.relationship1 = {'?':0, 'Wife':0, 'Own-child':0, 'Husband':0, 
			'Not-in-family':0, 'Other-relative':0, 'Unmarried':0}
		self.relationship2 = {'?':0, 'Wife':0, 'Own-child':0, 'Husband':0, 
			'Not-in-family':0, 'Other-relative':0, 'Unmarried':0}
	
		self.race1 = {'?':0, 'White':0, 'Asian-Pac-Islander':0, 
			'Amer-Indian-Eskimo':0, 'Other':0, 'Black':0}
		self.race2 = {'?':0, 'White':0, 'Asian-Pac-Islander':0, 
			'Amer-Indian-Eskimo':0, 'Other':0, 'Black':0}
	
		self.sex1 = {'?':0, 'Male':0, 'Female':0}
		self.sex2 = {'?':0, 'Male':0, 'Female':0}
	
		self.capital_gain1Disc = {}
		self.capital_gain2Disc = {}
		self.capital_gainCategories = []
	
		self.capital_loss1Disc = {}
		self.capital_loss2Disc = {}
		self.capital_lossCategories = []
	
		self.hours_per_week1Disc = {}
		self.hours_per_week2Disc = {}
		self.hours_per_weekCategories = []

	
		self.native_country1 = {'?':0, 'United-States':0, 'Cambodia':0, 'England':0, 
			'Puerto-Rico':0, 'Canada':0, 'Germany':0, 
			'Outlying-US(Guam-USVI-etc)':0, 'India':0, 'Japan':0, 
			'Greece':0, 'South':0, 'China':0, 'Cuba':0, 'Iran':0, 
			'Honduras':0, 'Philippines':0, 'Italy':0, 'Poland':0, 
			'Jamaica':0, 'Vietnam':0, 'Mexico':0, 'Portugal':0, 
			'Ireland':0, 'France':0, 'Dominican-Republic':0, 
			'Laos':0, 'Ecuador':0, 'Taiwan':0, 'Haiti':0, 
			'Columbia':0, 'Hungary':0, 'Guatemala':0, 'Nicaragua':0,
			'Scotland':0, 'Thailand':0, 'Yugoslavia':0, 'El-Salvador':0, 
			'Trinadad&Tobago':0, 'Peru':0, 'Hong':0, 'Holand-Netherlands':0}
		self.native_country2 ={'?':0, 'United-States':0, 'Cambodia':0, 'England':0, 
			'Puerto-Rico':0, 'Canada':0, 'Germany':0, 
			'Outlying-US(Guam-USVI-etc)':0, 'India':0, 'Japan':0, 
			'Greece':0, 'South':0, 'China':0, 'Cuba':0, 'Iran':0, 
			'Honduras':0, 'Philippines':0, 'Italy':0, 'Poland':0, 
			'Jamaica':0, 'Vietnam':0, 'Mexico':0, 'Portugal':0, 
			'Ireland':0, 'France':0, 'Dominican-Republic':0, 
			'Laos':0, 'Ecuador':0, 'Taiwan':0, 'Haiti':0, 
			'Columbia':0, 'Hungary':0, 'Guatemala':0, 'Nicaragua':0,
			'Scotland':0, 'Thailand':0, 'Yugoslavia':0, 'El-Salvador':0, 
			'Trinadad&Tobago':0, 'Peru':0, 'Hong':0, 'Holand-Netherlands':0}

		self.numOfClass1 = 0
		self.numOfClass2 = 0
		#print ("Model Constructor of type %s" % self.name)

	'''
	Method to load an adult object into our array of Adults
	'''
	def loadAdult(self, adult):
		self.Adults.append(adult)

	'''
	Method to load adult attributes into our Model for Preprocessing (Before Binning)
	'''
	def loadAdultAttributes(self, adult):
		if(adult.adultClass == ">50K"):
			self.numOfClass1 += 1
			self.age1[adult.age] += 1
			self.workclass1[adult.workclass] += 1
			self.fnlwgt1[adult.fnlwgt] += 1
			self.education1[adult.education] += 1
			self.education_num1[adult.education_num] += 1
			self.marital_status1[adult.marital_status] += 1
			self.occupation1[adult.occupation] += 1
			self.relationship1[adult.relationship] += 1
			self.race1[adult.race] += 1
			self.sex1[adult.sex] += 1
			self.capital_gain1[adult.capital_gain] += 1
			self.capital_loss1[adult.capital_loss] += 1
			self.hours_per_week1[adult.hours_per_week] += 1
			self.native_country1[adult.native_country] += 1
			return
		elif(adult.adultClass == "<=50K"):
			self.numOfClass2 += 1
			self.age2[adult.age] += 1
			self.workclass2[adult.workclass] += 1
			self.fnlwgt2[adult.fnlwgt] += 1
			self.education2[adult.education] += 1
			self.education_num2[adult.education_num] += 1
			self.marital_status2[adult.marital_status] += 1
			self.occupation2[adult.occupation] += 1
			self.relationship2[adult.relationship] += 1
			self.race2[adult.race] += 1
			self.sex2[adult.sex] += 1
			self.capital_gain2[adult.capital_gain] += 1
			self.capital_loss2[adult.capital_loss] += 1
			self.hours_per_week2[adult.hours_per_week] += 1
			self.native_country2[adult.native_country] += 1
			return

	'''
	Method to sort our Lists of Tuples for Continuous Attributes
	'''
	def sortContinuousLists(self):
		self.ages = sorted(self.ages, key=lambda tup:tup[0])
		self.fnlwgts = sorted(self.fnlwgts, key=lambda tup:tup[0])
		self.education_nums = sorted(self.education_nums, key=lambda tup:tup[0])
		self.capital_gains = sorted(self.capital_gains, key=lambda tup:tup[0])
		self.capital_losses = sorted(self.capital_losses, key=lambda tup:tup[0])
		self.hours_per_weeks = sorted(self.hours_per_weeks, key=lambda tup:tup[0])

	'''
	Method to read our dirty ARFF file and calculate various parameters, such as number
	of missing attributes, relations missing attributes, total attributes, number of 
	relations belonging to each class, etc, etc
	'''
	def ingestARFF(self, file):
		print("Ingesting ARFF with name: " + str(file))
		with open(file, 'r') as f:
			for line in f:
				if "@" in line:
					continue
				elif "@" not in line and line != "\n":
					if "?" in line:
						self.noRelationsMissing +=1
						self.noMissing += line.count('?')
					
					attr = line.split(", ")
					#These are the only attributes WITHOUT missing values
					if attr[0] == '?': print("found missing age")
					if attr[2] == '?': print("found missing fnlwgt")
					if attr[4] == '?': print("found missing edunum")
					if attr[10] == '?': print("found missing capgain")
					if attr[11] == '?': print("found missing caploss")
					if attr[12] == '?': print("found missing hoursperweek")

					#0 - age, 1 - workclass, 2 - fnlwgt, 3 - education, 4 - education_num, 5 - marital_status
					#6 - occupation, 7 - relationship, 8 - race, 9 - sex, 10 - capital_gain
					#11 - capital_loss, 12 - hours_per_week, 13 - native_country, 14 - income

					#Get rid of '\n' character for last item
					incomeClass = attr[14][:-1] 
					
					newAdult = Adult(attr[0], attr[1], attr[2], attr[3],
						 attr[4], attr[5], attr[6], attr[7], attr[8],
						 attr[9], attr[10], attr[11], attr[12], attr[13], attr[14][:-1])
					
					#Loads our Adult -> Model for Preprocessing
					self.loadAdultAttributes(newAdult)

					#add tuples to our lists for entropy-based discretization later
					self.ages.append((int(attr[0]), incomeClass))
					self.fnlwgts.append((int(attr[2]), incomeClass))
					self.education_nums.append((int(attr[4]), incomeClass))
					self.capital_gains.append((int(attr[10]), incomeClass))
					self.capital_losses.append((int(attr[11]), incomeClass))
					self.hours_per_weeks.append((int(attr[12]), incomeClass))

					self.noRelations += 1
			
			#print("There are %s attributes with missing values" % self.noMissing)
			#print("There are %s persons with at least 1 missing value" % self.noRelationsMissing)
			#print("There are %s relations in our dataset" % self.noRelations)
			#print("There are %s adults belonging to Class 1 (>50K)" % self.numOfClass1)
			#print("There are %s adults belonging to Class 2 (<=50K)" % self.numOfClass2)

	###########################################################################
	'''
	Methods for replacing missing values in ARFF.
	Model Object has list of Adult Objects.
	Iterate Through Each Adult Object, if item has ?, take the mode or average of its respective class depending on data type of attribute
	
	Grab Modes and Averages and place into dictionaries belonging to each class
	'''

	'''
	Calculates all modes for categorical attributes
	'''
	def calculateModes(self):
		self.workclass1mode = self.calculateMode(self.workclass1)
		self.workclass2mode = self.calculateMode(self.workclass2)
		self.education1mode = self.calculateMode(self.education1)
		self.education2mode = self.calculateMode(self.education2)
		self.marital_status1mode = self.calculateMode(self.marital_status1)
		self.marital_status2mode = self.calculateMode(self.marital_status2)
		self.occupation1mode = self.calculateMode(self.occupation1)
		self.occupation2mode = self.calculateMode(self.occupation2)
		self.relationship1mode = self.calculateMode(self.relationship1)
		self.relationship2mode = self.calculateMode(self.relationship2)
		self.race1mode = self.calculateMode(self.race1)
		self.race2mode = self.calculateMode(self.race2)
		self.sex1mode = self.calculateMode(self.sex1)
		self.sex2mode = self.calculateMode(self.sex2)
		self.native_country1mode = self.calculateMode(self.native_country1)
		self.native_country2mode = self.calculateMode(self.native_country2)

	'''
	Given a dictionary, find the mode
	'''
	def calculateMode(self, dictionary):
		largestFreq = 0
		Mode = ""
		for key, val in dictionary.iteritems():
			if dictionary[key] > largestFreq:
				largestFreq = dictionary[key]
				Mode = key
		return Mode

	'''
	Calculates all attribute averages for numerical attributes
	'''
	def calculateAverages(self):
		self.age1avg = self.calculateAvg(self.age1)
		self.age2avg = self.calculateAvg(self.age2)
		self.fnlwgt1avg = self.calculateAvg(self.fnlwgt1)
		self.fnlwgt2avg = self.calculateAvg(self.fnlwgt2)
		self.education_num1avg = self.calculateAvg(self.education_num1)
		self.education_num2avg = self.calculateAvg(self.education_num2)
		self.capital_gain1avg = self.calculateAvg(self.capital_gain1)
		self.capital_gain2avg = self.calculateAvg(self.capital_gain2)
		self.capital_loss1avg = self.calculateAvg(self.capital_loss1)
		self.capital_loss2avg = self.calculateAvg(self.capital_loss2)
		self.hours_per_week1avg = self.calculateAvg(self.hours_per_week1)
		self.hours_per_week2avg = self.calculateAvg(self.hours_per_week2)

	'''
	Given a defaultdict, find the average 
	'''
	def calculateAvg(self, dictionary):
		total = 0
		totalIndividuals = 0
		for key, val in dictionary.iteritems():
			if key == "?":
				continue
			total += dictionary[key]*int(key)
			totalIndividuals += dictionary[key]
		return int(total / totalIndividuals)

	'''
	Method to print all Modes and Averages of all ARFF Attributes
	'''
	def printModesAndAverages(self):
		print("Printing Modes and Averages")
		print("Class 1: >50K")
		print("="*30)
		print(self.age1avg, self.workclass1mode, self.fnlwgt1avg, self.education1mode, self.education_num1avg, self.relationship1mode,
			self.occupation1mode, self.marital_status1mode, self.race1mode, self.sex1mode, self.capital_gain1avg,
			self.capital_loss1avg, self.hours_per_week1avg, self.native_country1mode)
		print("Class 2: <=50K")
		print("="*30)
		print(self.age2avg, self.workclass2mode, self.fnlwgt2avg, self.education2mode, self.education_num2avg, self.relationship2mode,
			self.occupation2mode, self.marital_status2mode, self.race2mode, self.sex2mode, self.capital_gain2avg,
			self.capital_loss2avg, self.hours_per_week2avg, self.native_country2mode)

	'''
	Copies input data file and replaces all instances of '?' with mode/average of each respective class
	'''
	def replaceMissingAttributes(self, file):
		#print("Cleaning File with %s missing values" %self.noMissing)
		cleanedFile = open('clean-' + file, "w").writelines([l for l in open(file).readlines()]) 
	 	with open('clean-' + file, 'rw') as f:
	 		for line in fileinput.FileInput('clean-' + file, inplace=1):
	 			attributes = line.split(', ')
	 			for i in range(0, len(attributes)):
	 				if attributes[i] == '?':
	 					attributes[i] = self.findReplacementAttribute(i, attributes[14][:-1])
	 					self.noMissing -= 1
	 			attributes[len(attributes) - 1] = attributes[len(attributes) - 1][:-1]
	 			cleanedAttributes = ", ".join(attributes)
	 			print cleanedAttributes
	 	#print("Done Cleaning, there are now %s missing values" %self.noMissing)

	 	#Return the file name
	 	print("Returning File Name: %s" %('clean-' + str(file)))
	 	return ('clean-' + str(file))

	'''
	Method to increment dictionaries of Attribute:Frequency and return the average of our Model
	'''
	def findReplacementAttribute(self, col, classType):
		if classType == '>50K':
			if col == 0:
				self.age1[self.age1avg] += 1 
				return self.age1avg
			elif col == 1:
				self.workclass1[self.workclass1mode] += 1 
				return self.workclass1mode
			elif col == 2:
				self.fnlwgt1[self.fnlwgt1avg] += 1
				return self.fnlwgt1avg
			elif col == 3:
				self.education1[self.education1mode] += 1
				return self.education1mode
			elif col == 4:
				self.education_num1[self.education_num1avg] += 1
				return self.education_num1avg
			elif col == 5:
				self.relationship1[self.relationship1mode] += 1
				return self.relationship1mode
			elif col == 6:
				self.occupation1[self.occupation1mode] += 1
				return self.occupation1mode
			elif col == 7:
				self.marital_status1[self.marital_status1mode] += 1
				return self.marital_status1mode
			elif col == 8:
				self.race1[self.race1mode] += 1
				return self.race1mode
			elif col == 9:
				self.sex1[sex1mode] += 1
				return self.sex1mode
			elif col == 10:
				self.capital_gain1[self.capital_gain1avg] += 1
				return self.capital_gain1avg
			elif col == 11:
				self.capital_loss[self.capital_loss1avg] += 1
				return self.capital_loss1avg
			elif col == 12:
				self.hours_per_week1[self.hours_per_week1avg] += 1
				return self.hours_per_week1avg
			elif col == 13:
				self.native_country1[self.native_country1mode] += 1
				return self.native_country1mode
		elif classType == '<=50K':
			if col == 0: 
				self.age2[self.age2avg] += 1
				return self.age2avg
			elif col == 1: 
				self.workclass2[self.workclass2mode] += 1
				return self.workclass2mode
			elif col == 2:
				self.fnlwgt2[self.fnlwgt2avg] += 1
				return self.fnlwgt2avg
			elif col == 3:
				self.education2[self.education2mode] += 1
				return self.education2mode
			elif col == 4:
				self.education_num2[self.education_num2avg] += 1
				return self.education_num2avg
			elif col == 5:
				self.relationship2[self.relationship2mode] += 1
				return self.relationship2mode
			elif col == 6:
				self.occupation2[self.occupation2mode] += 1
				return self.occupation2mode
			elif col == 7:
				self.marital_status2[self.marital_status2mode] += 1
				return self.marital_status2mode
			elif col == 8:
				self.race2[self.race2mode] += 1
				return self.race2mode
			elif col == 9:
				self.sex2[self.sex2mode] += 1
				return self.sex2mode
			elif col == 10:
				self.capital_gain2[self.capital_gain2avg] += 1
				return self.capital_gain2avg
			elif col == 11:
				self.capital_loss2[self.capital_loss2avg] += 1
				return self.capital_loss2avg
			elif col == 12:
				self.hours_per_week2[self.hours_per_week2avg] += 1
				return self.hours_per_week2avg
			elif col == 13:
				self.native_country2[self.native_country2mode] += 1
				return self.native_country2mode

	###########################################################################			
	'''
	DISCRETIZATION METHODS 
	'''
	'''
	Takes in a sorted list of 2-value tuples (Attr, Class) and should return a list of bins
	#Given a list of 2-value tuples, what do we need to do?
		#1) Calculate the Big Entropy Value that we need to compare our binning to
		
		#2) Iterate through our list of values, calculating the:
		#	Size of both bins,
		#	Bin labels (average of upper num of bin 1, lower num of bin 2)
		#	How many items in bin 1 fall in C1 and C2
		#	How many items in bin 2 fall in C1 and C2
		#3) With our numbers, we should be able to calculate the entropy
		#4) Find the best entropy, split the bins, add the category-label
		#5) Repeat until we get our sufficient # of bins

	TODO: Allow for N-Bins, Not just 4! This is hella jank
	'''

	def discretizeAttribute(self, attributes):
		#print("Discretizing")
		#No. of Bins
		BINS = 4
		binstotal = 1
		indices = []
		categories = []
		#For Loop Here with limit as # of bins
		leftside = None 
		rightside = None 
		index = None

		while binstotal < BINS:
			leftside, rightside, index, category = self.findBestSplit(attributes)
			indices.append(index)
			categories.append(category)
			llside, lrside, secondind, category = self.findBestSplit(leftside)
			binstotal += 1
			categories.insert(0, category)
			rlside, rrside, thirdind, category = self.findBestSplit(rightside)
			binstotal += 1
			categories.append(category)
			indices.insert(0, secondind)
			indices.append(thirdind + index)
			binstotal += 1

		indices = sorted(indices)
		#print categories
		categories = self.processCategories(categories)
		#print categories
		return categories

	'''
	Method to iterate through our clean file, and count the times each value falls into a category 
	'''
	def loadDiscretizedAttributes(self, file):
		count = 0
		with open(file, 'r') as f:
			for line in f:
				if "@" in line:
					continue
				elif "@" not in line and line != "\n":
					attr = line.split(", ")
					#0 - age, 1 - workclass, 2 - fnlwgt, 3 - education, 4 - education_num, 5 - marital_status
					#6 - occupation, 7 - relationship, 8 - race, 9 - sex, 10 - capital_gain
					#11 - capital_loss, 12 - hours_per_week, 13 - native_country, 14 - income
					incomeClass = attr[14][:-1] 
					
					self.loadDiscretizedValue(attr[0], incomeClass, self.age1Disc, self.age2Disc, self.ageCategories)
					self.loadDiscretizedValue(attr[2], incomeClass, self.fnlwgt1Disc, self.fnlwgt2Disc, self.fnlwgtCategories)
					self.loadDiscretizedValue(attr[4], incomeClass, self.education_num1Disc, self.education_num2Disc, self.education_numCategories)
					self.loadDiscretizedValue(attr[10], incomeClass, self.capital_gain1Disc, self.capital_gain2Disc, self.capital_gainCategories)
					self.loadDiscretizedValue(attr[11], incomeClass, self.capital_loss1Disc, self.capital_loss2Disc, self.capital_lossCategories)
					self.loadDiscretizedValue(attr[12], incomeClass, self.hours_per_week1Disc, self.hours_per_week2Disc, self.hours_per_weekCategories)
					count += 1

		#print ("We've got to load %d" %(count))

	'''
	#Mutates the Dictionary Value, for each returned bin from findCategory()
	'''
	def loadDiscretizedValue(self, value, income, dictOne, dictTwo, categories):
		bin = self.findCategory(value, categories)
		if income == "<=50K":
			if bin in dictOne.keys():
				dictOne[bin] += 1
			else:
				dictOne[bin] = 1
		elif income == ">50K":
			if bin in dictTwo.keys():
				dictTwo[bin] += 1
			else:
				dictTwo[bin] = 1
	'''
	Method that finds which Category a value belongs to, and returns the bin label
	'''
	def findCategory(self, value, categories):
		value = int(value)
		for item in categories:
			if item[0] == "X":
				valToCompare = int(item[3:])
				if value <= valToCompare:
					#print (value, "should be in bin %s" %item)
					return item
			elif item[len(item)-1] == "X":
				valToCompare = int(item.split("<X")[0])
				if value > valToCompare: 
					#print (value, "should be in bin %s" %item)
					return item
			else:
				bounds = item.split("<X<=")
				lowerB = int(bounds[0])
				upperB = int(bounds[1])
				if value > lowerB and value <= upperB:
					#print (value, "lies between %d and %d" %(lowerB, upperB))
					#print "return category %s" %item
					return item
	'''
	Method to process our selected attributes for splitting -> categories we will discretize with
	Takes in a list in format = 
	'''
	def processCategories(self, categories):
		categoryLabels = []

		firstItem = True
		prevItem = ""
		valueHolder = "X"
		for item in categories:
			if (firstItem):
				if item[:1] == "<":
					label = valueHolder + "<=" + str(int(item[1:]) - 1)
					firstItem = False
					categoryLabels.append(label)
					prevItem = str(int(item[1:]) - 1)
				elif item[:1] == ">":
					label = valueHolder + "<=" + item[1:]
					firstItem = False
					categoryLabels.append(label)
					prevItem = item[1:]
			elif not (firstItem):
				if item[:1] == "<":
					label = prevItem + "<" + valueHolder + "<=" + str(int(item[1:]) - 1)
					categoryLabels.append(label)
					prevItem = str(int(item[1:]) - 1)
				elif item[:1] == ">":
					label = prevItem + "<" + valueHolder + "<=" + item[1:]
					categoryLabels.append(label)
					prevItem = item[1:]

		categoryLabels.append(prevItem + "<" + valueHolder)

		#print("Category Labels: ", categoryLabels)
		return categoryLabels

	'''
	Method to find the most optimal split given a sorted list of 2-Value Tuples (val, class)
	'''
	def findBestSplit(self, attributes):
		#print("Calculating Entropy of Unsplit List of Attributes")
		noClass1 = float(0)
		noClass2 = float(0)
		size = float(len(attributes))
		#Initial calculation of # of classes in our data set
		for item in attributes:
			if item[1] == ">50K": 
				noClass1 += 1
			elif item[1] == "<=50K":
				noClass2 += 1
		largeEntropy = self.calculateEntropy(float(noClass1), float(noClass2), float(size))
		#The entropy of the data set that we need to subtract our partition's entropy from
		#print("We need to beat %f!" %largeEntropy)
		#How man Classes occur on the right split of the index
		RHSClass1 = float(noClass1)
		RHSClass2 = float(noClass2)
		noClass1 = float(0)
		noClass2 = float(0)
		#Our Worst Entropy Score, Grab the lowest entropy score that falls below this
		#Should never be more than 1
		ent = float(1.0)
		index = 0
		for i in range(0, int(size-1)):
			#print("Finding %dth Entropy for Split between %d and %d" % (i+1, attributes[i][0], attributes[i+1][0]))
			if attributes[i][1] == ">50K":
				noClass1 += 1
			elif attributes[i][1] == "<=50K":
				noClass2 += 1
			#Left Side Contribution
			a = self.calculateEntropy(float(noClass1), float(noClass2), float(i+1))
			#Right Side
			b = self.calculateEntropy(float(RHSClass1 - noClass1), float(RHSClass2 - noClass2), float(size - i))
			#Result
			res = float(i+1)/float(size) * a + float(size - i)/float(size) * b
			if res <= ent: 
				ent = float(res)
				index = i

		if attributes[index-1][0] == attributes[index][0] or attributes[index][0] == attributes[index+1][0]:
			#print ("Oh shoot, looks like we need to slide down and find some index that better splits our attribute!")
			index, category = self.findClosestDifferentAttribute(attributes, index)
		#print (ent)
		#print ("Our Best Split is at index: %s with entropy score of %f"  %(index, float(ent)))
		return attributes[:index], attributes[index:], index, category
		
	'''
	Method to find closest index of differing attribute so we can categorize our stuff
	Start from middle, explore L and R until you hit a different value
	'''
	def findClosestDifferentAttribute(self, attributes, index):
		tempBool = False
		attr = attributes[index][0]
		category = ""
		#print ("Attribute we are stuck with: %s. Indexed at %d"  %(attr, index))
		newIndex = 0
		i = 0
		while tempBool == False:
			#Grab all items >= attr
			if attributes[index+i][0] != attr:
				#print("Hit the right first")
				newIndex = index+i
				category = "<" + str(attributes[index+i][0])
				tempBool = True 
			#Grab all items < attr
			elif attributes[index-i][0] != attr:
				#print("Hit the left first")
				newIndex = index-i
				category = ">" + str(attributes[index-i][0])
				tempBool = True
			i += 1
			#end while loop

		#print("Our new index is %d" % newIndex)
		#print("Our new attribute value is %s" % attributes[newIndex][0])
		#print("Category is now: %s" % category)
		#Return category
		return newIndex, category

	'''
	Method to Calculate Entropy of Data Set, given number of occurences of each class, and size
	'''
	def calculateEntropy(self, noClass1, noClass2, size):
		#if noClass1 is still zero, domain of log must be > 0
		if noClass1 != 0.0:
			class1Contrib = (noClass1/size) * math.log(noClass1/size)
		else:
			class1Contrib = 0.0

		if noClass2 != 0.0:
			class2Contrib = (noClass2/size) * math.log(noClass2/size)
		else:
			class2Contrib = 0.0
		
		entropy = float(-1 * (class1Contrib + class2Contrib))
		return entropy
	
	'''
	Method to replace all continuous attributes with their categorical labels 
	====================================================
	=
	=
	=
		#0 - age, 1 - workclass, 2 - fnlwgt, 3 - education, 4 - education_num, 5 - marital_status
		#6 - occupation, 7 - relationship, 8 - race, 9 - sex, 10 - capital_gain	
		#11 - capital_loss, 12 - hours_per_week, 13 - native_country, 14 - income
	=
	=
	'''
	def categorizeContinuousAttributes(self, file):
		#print("Performing Binning on File")
		binnedFile = open('binned-' + file, "w").writelines([l for l in open(file).readlines()]) 
	 	with open('binned-' + file, 'rw') as f:
	 		#For each line
	 		for line in fileinput.FileInput('binned-' + file, inplace=1):
	 			attributes = line.split(', ')
	 			'''
				FIX LOGIC HERE SO THAT WE REPLACE CONTINUOUS VALUES WITH BIN LABELS
				'''
				if "@" not in line and line != "\n":
					for i in range(0, len(attributes)):
						if i == 0:
							attributes[i] = self.findCategory(attributes[0], self.ageCategories)
						elif i == 2:
							attributes[i] = self.findCategory(attributes[2], self.fnlwgtCategories)
						elif i == 4:
							attributes[i] = self.findCategory(attributes[4], self.education_numCategories)
						elif i == 10:
							attributes[i] = self.findCategory(attributes[10], self.capital_gainCategories)
						elif i == 11:
							attributes[i] = self.findCategory(attributes[11], self.capital_lossCategories)
						elif i == 12:
							attributes[i] = self.findCategory(attributes[12], self.hours_per_weekCategories)

					newAdult = Adult(attributes[0], attributes[1], attributes[2], attributes[3],
						 attributes[4], attributes[5], attributes[6], attributes[7], attributes[8],
						 attributes[9], attributes[10], attributes[11], attributes[12], attributes[13], attributes[14][:-1])
					'''Loads the Adult with Categorized Continuous Values into our list of Adults'''
					self.loadAdult(newAdult)				
				#Prevent adding a second new line
				attributes[len(attributes) - 1] = attributes[len(attributes) - 1][:-1]
				cleanedAttributes = ", ".join(attributes)
				print cleanedAttributes
	 	#print("Done Categorizing")
	 	#Return the file name
	 	print("Returning File Name: %s" %('binned-' + str(file)))
	 	return ('binned-' + str(file))

	'''
	Method that iterates through our Adults List and prints its attributes.
	This will never print the binned or replaced attributes because we don't bother
	to replace the characteristics of the Adult Object as we replace :(
	'''
	def printRelations(self):
		#print(self.noRelations)		
		for i in range(0, len(self.Adults)):
			self.Adults[i].printAttributes()

	'''
	Naive Method to print all information about our model, maybe move this somewhere else...
	'''
	def printModel(self):
		print("Printing Model for %s" %(self.name))
		###########################################################################
		#Class 1
		###########################################################################
		print("="*50 + '\n\n' + "Printing Model for Class 1: >50K" + '\n\n' + "="*50)
		print("Number of individuals belonging to Class: %s \n\n" %self.numOfClass1)
		print("Age:Frequency")
		print(self.age1Disc)
		print("Working Class:Frequency")
		print(self.workclass1)
		print("Education:Frequency")
		print(self.education1)
		print("Education Num:Frequency")
		print(self.education_num1Disc)
		print("Marital Status:Frequency")
		print(self.marital_status1)
		print("Occupation:Frequency")
		print(self.occupation1)
		print("Relationship:Frequency") 
		print(self.relationship1)
		print("Race:Frequency")
		print(self.race1)
		print("Sex:Frequency")
		print(self.sex1)
		print("Capital_Gain:Frequency")
		print(self.capital_gain1Disc)
		print("Capital_Loss:Frequency")
		print(self.capital_loss1Disc)
		print("Hours_Per_Week:Frequency")
		print(self.hours_per_week1Disc)
		print("Native Country:Frequency")
		print(self.native_country1)
		###########################################################################
		#Class 2
		###########################################################################
		print("="*50 + '\n\n' + "Printing Model for Class 2: <=50K" + '\n\n' + "="*50)
		print("Number of individuals belonging to Class: %s \n\n" %self.numOfClass2)
		print("Age:Frequency")
		print(self.age2Disc)
		print("Working Class:Frequency")
		print(self.workclass2)
		print("Education:Frequency")
		print(self.education2)
		print("Education Num:Frequency")
		print(self.education_num2Disc)
		print("Marital Status:Frequency")
		print(self.marital_status2)
		print("Occupation:Frequency")
		print(self.occupation2)
		print("Relationship:Frequency") 
		print(self.relationship2)
		print("Race:Frequency")
		print(self.race2)
		print("Sex:Frequency")
		print(self.sex2)
		print("Capital_Gain:Frequency")
		print(self.capital_gain2Disc)
		print("Capital_Loss:Frequency")
		print(self.capital_loss2Disc)
		print("Hours_Per_Week:Frequency")
		print(self.hours_per_week2Disc)
		print("Native Country:Frequency")
		print(self.native_country2)

		print("This model was created using %d adults" %(self.numOfClass1 + self.numOfClass2))
		###########################################################################

	'''


	TRAINING METHODS, Should involve copying over a list of Adults, sorting into AdultsTest and AdultsTrain
	Iterating through the lists of AdultsTrain and updating the dictionaries
	Calculating
	

	'''

	def partitionAdults(self, modelNum):
		totalNumberOfAdults = len(self.Adults)
		adultsToTrain = int(9*totalNumberOfAdults / 10)
		adultsToTest = totalNumberOfAdults - adultsToTrain

		indexStart = (modelNum - 1) * adultsToTest
		indexEnd = (modelNum - 0) * adultsToTest - 1
		#print("Our Starting Index and Ending Index")
		#print indexStart, indexEnd

		for i in range(0, totalNumberOfAdults):
			if i >= indexStart and i < indexEnd:
				self.AdultsTest.append(self.Adults[i])
			else:
				self.AdultsTrain.append(self.Adults[i])

		#print("We added %d adults for training and %d adults for testing" %(len(self.AdultsTrain), len(self.AdultsTest)))
		#print("First Item in Testing Data")
		#self.AdultsTest[0].printAttributes()
		#print("First Item in Training Data")
		#self.AdultsTrain[0].printAttributes()

	def trainModel(self):
		print ("Training model with %d adults" %len(self.AdultsTrain))
		for adult in self.AdultsTrain:
			if adult.adultClass == ">50K":
				self.numOfClass1 += 1
				if adult.age in self.age1Disc.keys(): self.age1Disc[adult.age] += 1
				else: self.age1Disc[adult.age] = 1
				if adult.workclass in self.workclass1.keys(): self.workclass1[adult.workclass] += 1
				else: self.workclass1[adult.workclass] = 1
				if adult.fnlwgt in self.fnlwgt1Disc.keys(): self.fnlwgt1Disc[adult.fnlwgt] += 1
				else: self.fnlwgt1Disc[adult.fnlwgt] = 1
				if adult.education in self.education1.keys(): self.education1[adult.education] += 1
				else: self.education1[adult.education] = 1
				if adult.education_num in self.education_num1Disc.keys(): self.education_num1Disc[adult.education_num] += 1
				else: self.education_num1Disc[adult.education_num] = 1
				if adult.marital_status in self.marital_status1.keys(): self.marital_status1[adult.marital_status] += 1
				else: self.marital_status1[adult.marital_status] = 1
				if adult.occupation in self.occupation1.keys(): self.occupation1[adult.occupation] += 1
				else: self.occupation1[adult.occupation] = 1
				if adult.relationship in self.relationship1.keys(): self.relationship1[adult.relationship] += 1
				else: self.relationship1[adult.relationship] = 1
				if adult.race in self.race1.keys(): self.race1[adult.race] += 1
				else: self.race1[adult.race] = 1
				if adult.sex in self.sex1.keys(): self.sex1[adult.sex] += 1
				else: self.sex1[adult.sex] = 1
				if adult.capital_gain in self.capital_gain1Disc.keys(): self.capital_gain1Disc[adult.capital_gain] += 1
				else: self.capital_gain1Disc[adult.capital_gain] = 1
				if adult.capital_loss in self.capital_loss1Disc.keys(): self.capital_loss1Disc[adult.capital_loss] += 1
				else: self.capital_loss1Disc[adult.capital_loss] = 1
				if adult.hours_per_week in self.hours_per_week1Disc.keys(): self.hours_per_week1Disc[adult.hours_per_week] += 1
				else: self.hours_per_week1Disc[adult.hours_per_week] = 1
				if adult.native_country in self.native_country1.keys(): self.native_country1[adult.native_country] += 1
				else: self.native_country1[adult.native_country] = 1
			elif adult.adultClass == "<=50K":
				self.numOfClass2 += 1
				if adult.age in self.age2Disc.keys(): self.age2Disc[adult.age] += 1
				else: self.age2Disc[adult.age] = 1
				if adult.workclass in self.workclass2.keys(): self.workclass2[adult.workclass] += 1
				else: self.workclass2[adult.workclass] = 1
				if adult.fnlwgt in self.fnlwgt2Disc.keys(): self.fnlwgt2Disc[adult.fnlwgt] += 1
				else: self.fnlwgt2Disc[adult.fnlwgt] = 1
				if adult.education in self.education2.keys(): self.education2[adult.education] += 1
				else: self.education2[adult.education] = 1
				if adult.education_num in self.education_num2Disc.keys(): self.education_num2Disc[adult.education_num] += 1
				else: self.education_num2Disc[adult.education_num] = 1
				if adult.marital_status in self.marital_status2.keys(): self.marital_status2[adult.marital_status] += 1
				else: self.marital_status2[adult.marital_status] = 1
				if adult.occupation in self.occupation2.keys(): self.occupation2[adult.occupation] += 1
				else: self.occupation2[adult.occupation] = 1
				if adult.relationship in self.relationship2.keys(): self.relationship2[adult.relationship] += 1
				else: self.relationship2[adult.relationship] = 1
				if adult.race in self.race2.keys(): self.race2[adult.race] += 1
				else: self.race2[adult.race] = 1
				if adult.sex in self.sex2.keys(): self.sex2[adult.sex] += 1
				else: self.sex2[adult.sex] = 1
				if adult.capital_gain in self.capital_gain2Disc.keys(): self.capital_gain2Disc[adult.capital_gain] += 1
				else: self.capital_gain2Disc[adult.capital_gain] = 1
				if adult.capital_loss in self.capital_loss2Disc.keys(): self.capital_loss2Disc[adult.capital_loss] += 1
				else: self.capital_loss2Disc[adult.capital_loss] = 1
				if adult.hours_per_week in self.hours_per_week2Disc.keys(): self.hours_per_week2Disc[adult.hours_per_week] += 1
				else: self.hours_per_week2Disc[adult.hours_per_week] = 1
				if adult.native_country in self.native_country2.keys(): self.native_country2[adult.native_country] += 1
				else: self.native_country2[adult.native_country] = 1
		#Need to include all keys in both dicts
		for key in self.capital_gain1Disc.keys():
			if key not in self.capital_gain2Disc.keys():
				self.capital_gain2Disc[key] = 0

	'''
	Method to Test our Model:
	Go through every adult in AdultsTest list, and calculate the probability that it is Class 1 and Class 2:
	Class 1 - >50K
	Class 2 - <=50K


	An example probability for an Adult with attributes X: < a1, a2, a3..., an >:


	Returns all validation quantities
	'''
	def testModel(self):
		print("Testing %s with %d adults" %(self.name, len(self.AdultsTest)))
		sizeOfSet = len(self.AdultsTrain)
		print("Size of Training Set: %d, Number of Class 1: %d,  Number of Class 2: %d" %(sizeOfSet, self.numOfClass1, self.numOfClass2))
		i=0
		#Need TP1, TP2, FP1, FP2, TN1, TN2, FN1, FN2, increment two for each case
		TP1 = 0
		TP2 = 0
		FP1 = 0
		FP2 = 0
		TN1 = 0
		TN2 = 0
		FN1 = 0
		FN2 = 0
		for adult in self.AdultsTest:
			p1 = Decimal(self.calculateProbabilityIsClass1(adult, sizeOfSet))
			p2 = Decimal(self.calculateProbabilityIsClass2(adult, sizeOfSet))
			if p1 > p2:
				if adult.adultClass == ">50K": 
					TP1 += 1 
					TN2 += 1
				else: 
					FP1 += 1
					FN2 += 1
			elif p2 > p1: 
				if adult.adultClass == "<=50K":
				 	TN1 += 1
				 	TP2 += 1
				else: 
					FN1 += 1
					FP2 += 1
			i+=1
		print ("Accuracy for %s is: " %(self.name))
		accuracy = float(TP1 + TN1) / float(TP1 + TN1 + FP1 + FN1)
		print (accuracy)

		print ("Macro Precision for %s is: " %(self.name))
		precision1 = float(TP1) / float(TP1 + FP1)
		precision2 = float(TP2) / float(TP2 + FP2)
		macroPrecision = float(precision1 + precision2) / (float(2)) 
		print (macroPrecision)
		print ("Micro Precision for %s is: " %(self.name))
		microPrecision = float(TP1 + TP2) / float(TP1 + TP2 + FP1 + FP2)
		print (microPrecision)

		print ("Macro Recall/Sensitivity for %s is: " %(self.name))
		sensitivity1 = float(TP1) / float(TP1 + FN1)
		sensitivity2 = float(TP2) / float(TP2 + FN2)
		macroSensitivity = float(sensitivity1 + sensitivity2) / float(2)
		print (macroSensitivity)
		print ("Micro Sensitivity for %s is: " %(self.name))
		microSensitivity = float(TP1 + TP2) / float(TP1 + TP2 + FN1 + FN2)
		print (microSensitivity)

		print ("Macro Specificity for %s is: " %(self.name))
		specificity1 = float(TN1) / float(FP1 + TN1)
		specificity2 = float(TN2) / float(FP2 + TN2)
		macroSpecificity = float(specificity1 + specificity2) / float(2)
		print (macroSpecificity)
		print ("Micro Specificity for %s is: " %(self.name))
		microSpecificity = float(TN1 + TN2) / float(FP1 + FP2 + TN1 + TN2)
		print (microSpecificity)

		print ("Macro F1 Measure for %s is: " %(self.name))
		f1measure1 = float(2*sensitivity1 * precision1) / float(sensitivity1 + precision1)
		f1measure2 = float(2*sensitivity2 * precision2) / float(sensitivity2 + precision2)
		macroF1Measure = float(f1measure1 + f1measure2) / float(2)
		microF1Measure = float(2*microPrecision * microSensitivity) / float(microPrecision + microSensitivity)
		print (macroF1Measure)
		print ("Micro F1 Measure for %s is: " %(self.name))
		print (microF1Measure)
		return microPrecision, microSensitivity, microF1Measure, macroPrecision, macroSensitivity, macroF1Measure, accuracy

	'''
	Method to calculate the probability of an adult existing in Class 1 (>50K)
	'''
	def calculateProbabilityIsClass1(self, adult, sizeOfSet):
		probOfClass1 = float(self.numOfClass1)/float(sizeOfSet)
		ageProb = float(self.age1Disc[adult.age])/float(self.numOfClass1)
		workclassProb = float(self.workclass1[adult.workclass])/float(self.numOfClass1)
		fnlwgtProb = float(self.fnlwgt1Disc[adult.fnlwgt])/float(self.numOfClass1)
		educationProb = float(self.education1[adult.education])/float(self.numOfClass1)
		education_numProb = float(self.education_num1Disc[adult.education_num])/float(self.numOfClass1)
		marital_statusProb = float(self.marital_status1[adult.marital_status])/float(self.numOfClass1)
		occupationProb = float(self.occupation1[adult.occupation])/float(self.numOfClass1)
		relationshipProb = float(self.relationship1[adult.relationship])/float(self.numOfClass1)
		raceProb = float(self.race1[adult.race])/float(self.numOfClass1)
		sexProb = float(self.sex1[adult.sex])/float(self.numOfClass1)
		capital_gainProb = float(self.capital_gain1Disc[adult.capital_gain])/float(self.numOfClass1)
		capital_lossProb = float(self.capital_loss1Disc[adult.capital_loss])/float(self.numOfClass1)
		hours_per_weekProb = float(self.hours_per_week1Disc[adult.hours_per_week])/float(self.numOfClass1)
		native_countryProb = float(self.native_country1[adult.native_country])/float(self.numOfClass1)

		aP = Decimal(ageProb)
		wcP = Decimal(workclassProb)
		fwP = Decimal(fnlwgtProb)
		eP = Decimal(educationProb)
		enP = Decimal(education_numProb)
		msP = Decimal(marital_statusProb)
		oP = Decimal(occupationProb)
		reP = Decimal(relationshipProb)
		sP = Decimal(sexProb)
		cgP = Decimal(capital_gainProb)
		clP = Decimal(capital_lossProb)
		hpwP = Decimal(hours_per_weekProb)
		ncP = Decimal(native_countryProb)

		result = Decimal(probOfClass1*ageProb*workclassProb* \
				educationProb*education_numProb* \
				marital_statusProb*occupationProb*relationshipProb*raceProb*sexProb*capital_gainProb* \
				capital_lossProb*hours_per_weekProb*native_countryProb)

		res = aP * wcP * eP * enP * msP * oP * reP * sP * cgP * clP * hpwP * ncP
		return Decimal(res)

	'''
	Method to calculate the probability of an adult existing in Class 2 (<=50K)
	'''
	def calculateProbabilityIsClass2(self, adult, sizeOfSet):
		probOfClass2 = float(self.numOfClass2)/float(sizeOfSet)
		ageProb = float(self.age2Disc[adult.age])/float(self.numOfClass2)
		workclassProb = float(self.workclass2[adult.workclass])/float(self.numOfClass2)
		fnlwgtProb = float(self.fnlwgt2Disc[adult.fnlwgt])/float(self.numOfClass2)
		educationProb = float(self.education2[adult.education])/float(self.numOfClass2)
		education_numProb = float(self.education_num2Disc[adult.education_num])/float(self.numOfClass2)
		marital_statusProb = float(self.marital_status2[adult.marital_status])/float(self.numOfClass2)
		occupationProb = float(self.occupation2[adult.occupation])/float(self.numOfClass2)
		relationshipProb = float(self.relationship2[adult.relationship])/float(self.numOfClass2)
		raceProb = float(self.race2[adult.race])/float(self.numOfClass2)
		sexProb = float(self.sex2[adult.sex])/float(self.numOfClass2)
		capital_gainProb = float(self.capital_gain2Disc[adult.capital_gain])/float(self.numOfClass2)
		capital_lossProb = float(self.capital_loss2Disc[adult.capital_loss])/float(self.numOfClass2)
		hours_per_weekProb = float(self.hours_per_week2Disc[adult.hours_per_week])/float(self.numOfClass2)
		native_countryProb = float(self.native_country2[adult.native_country])/float(self.numOfClass2)

		aP = Decimal(ageProb)
		wcP = Decimal(workclassProb)
		fwP = Decimal(fnlwgtProb)
		eP = Decimal(educationProb)
		enP = Decimal(education_numProb)
		msP = Decimal(marital_statusProb)
		oP = Decimal(occupationProb)
		reP = Decimal(relationshipProb)
		sP = Decimal(sexProb)
		cgP = Decimal(capital_gainProb)
		clP = Decimal(capital_lossProb)
		hpwP = Decimal(hours_per_weekProb)
		ncP = Decimal(native_countryProb)

		result = Decimal(probOfClass2*ageProb*workclassProb* \
				educationProb*education_numProb* \
				marital_statusProb*occupationProb*relationshipProb*raceProb*sexProb*capital_gainProb* \
				capital_lossProb*hours_per_weekProb*native_countryProb)

		res = aP * wcP * eP * enP * msP * oP * reP * sP * cgP * clP * hpwP * ncP
		return Decimal(res)

'''
End of Model Class
'''

def writeResults(m1res, m2res, m3res, m4res, m5res, m6res, m7res, m8res, m9res, m10res, averages):
		fileName = "adults.out"
		headers = ["MicroPrecision","MicroRecall","Micro F1","MacroPrecision","MacroRecall","Macro F1","Accuracy"]
		models = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6", "Model 7", "Model 8", "Model 9", "Model 10"]
		print ("writing results to file: %s" %(fileName))
		with open(fileName, 'w') as f:
			f.write("Naive Bayes Classifier Validation Results" + "\n")
			header = "\t\t"
			for item in headers:
				header += (item + '\t')
			f.write(header + "\n")
			print(header + "\n")
			line = models[0] + '\t\t'
			for result in m1res:
				line += (str(result) + '\t')
			f.write(line + "\n")
			print(line + "\n")
			line = models[1] + '\t\t'
			for result in m2res:
				line += (str(result) + '\t')
			f.write(line + "\n")
			print(line + "\n")
			line = models[2] + '\t\t'
			for result in m3res:
				line += (str(result) + '\t')
			f.write(line + "\n")
			print(line + "\n")
			line = models[3] + '\t\t'
			for result in m4res:
				line += (str(result) + '\t')
			f.write(line + "\n")
			print(line + "\n")
			line = models[4] + '\t\t'
			for result in m5res:
				line += (str(result) + '\t')
			f.write(line + "\n")
			print(line + "\n")
			line = models[5] + '\t\t'
			for result in m6res:
				line += (str(result) + '\t')
			f.write(line + "\n")
			print(line + "\n")
			line = models[6] + '\t\t'
			for result in m7res:
				line += (str(result) + '\t')
			f.write(line + "\n")
			print(line + "\n")
			line = models[7] + '\t\t'
			for result in m8res:
				line += (str(result) + '\t')
			f.write(line + "\n")
			print(line + "\n")
			line = models[8] + '\t\t'
			for result in m9res:
				line += (str(result) + '\t')
			f.write(line + "\n")
			print(line + "\n")
			line = models[9] + '\t'
			for result in m10res:
				line += (str(result) + '\t')
			f.write(line + "\n")
			print(line + "\n")
			line = "Averages: " + '\t'
			for result in averages:
				line += (str(result) + '\t')
			f.write(line + "\n")
			print(line + "\n")
			f.close()

if __name__ == "__main__":
	'''CONSTANTS'''
	#print sys.argv[0]
	FILE_NAME = ''
	if len(sys.argv) > 1:
		FILE_NAME = sys.argv[1]
	else:
		FILE_NAME = "adult-big.arff"
	HEADER_SIZE = 19
	START_TIME = timeit.default_timer()

	'''Setting up Decimal Context'''
	getcontext().prec = 15

	'''
	BUILDING THE INITIAL MODEL FOR PREPROCESSING
	'''
	NBModel = Model("0")	
	NBModel.ingestARFF(FILE_NAME)
	NBModel.calculateModes()
	NBModel.calculateAverages()
	
	'''
	CLEANING THE FILE AND REPLACING THE MISSING VALUES WITH MODES AND AVERAGES
	'''
	CLEANED_FILE_NAME = NBModel.replaceMissingAttributes(FILE_NAME)
	
	'''Sort for Entropy Calculations'''
	NBModel.sortContinuousLists()

	'''Returns Categories that we would like to discretize our relations' ages with'''
	NBModel.ageCategories = NBModel.discretizeAttribute(NBModel.ages)
	'''Returns Categories that we would like to discretize our relations' fnlwgts with'''
	NBModel.fnlwgtCategories = NBModel.discretizeAttribute(NBModel.fnlwgts)
	'''Returns Categories that we would like to discretize our relation's education nums with'''
	NBModel.education_numCategories = NBModel.discretizeAttribute(NBModel.education_nums)
	'''Returns Categories that we would like to discretize our relation's capital gains with'''
	NBModel.capital_gainCategories = NBModel.discretizeAttribute(NBModel.capital_gains)
	'''Returns Categories that we would like to discretize our relation's capital_losses with'''
	NBModel.capital_lossCategories = NBModel.discretizeAttribute(NBModel.capital_losses)
	'''Returns Categories that we would like to discretize our relation's hours per week with'''
	NBModel.hours_per_weekCategories = NBModel.discretizeAttribute(NBModel.hours_per_weeks)

	'''Load Discretized Attributes Into Our Model'''
	NBModel.loadDiscretizedAttributes(CLEANED_FILE_NAME)
	''' return the name of the binned-clean ARFF '''
	BINNED_CLEANED_FILE_NAME = NBModel.categorizeContinuousAttributes(CLEANED_FILE_NAME)
	NBModel.printModel()
	'''
	At this point, we now have properly binned our values into binned-clean-adult-big.arff. So now we need to train 10 models, 
	each taking 90 percent of data set, and testing with the remaining 10 percent. Taking the ten percent, we can calculate accuracy
	and post them numbers,ya
	'''
	#NBModel.printRelations()

	'''Our 10 Models we will use to cross validate'''
	NBModel1 = Model("1")
	NBModel1.Adults = NBModel.Adults
	NBModel1.partitionAdults(1)
	NBModel1.trainModel()
	#NBModel1.printModel()
	model1res = NBModel1.testModel()
	
	NBModel2 = Model("2")
	NBModel2.Adults = NBModel.Adults
	NBModel2.partitionAdults(2)
	NBModel2.trainModel()
	#NBModel2.printModel()
	model2res = NBModel2.testModel()

	NBModel3 = Model("3")
	NBModel3.Adults = NBModel.Adults
	NBModel3.partitionAdults(3)
	NBModel3.trainModel()
	#NBModel3.printModel()
	model3res = NBModel3.testModel()

	NBModel4 = Model("4")
	NBModel4.Adults = NBModel.Adults
	NBModel4.partitionAdults(4)
	NBModel4.trainModel()
	#NBModel4.printModel()
	model4res = NBModel4.testModel()

	NBModel5 = Model("5")
	NBModel5.Adults = NBModel.Adults
	NBModel5.partitionAdults(5)
	NBModel5.trainModel()
	#NBModel5.printModel()
	model5res = NBModel5.testModel()

	NBModel6 = Model("6")
	NBModel6.Adults = NBModel.Adults
	NBModel6.partitionAdults(6)
	NBModel6.trainModel()
	#NBModel6.printModel()
	model6res = NBModel6.testModel()

	NBModel7 = Model("7")
	NBModel7.Adults = NBModel.Adults
	NBModel7.partitionAdults(7)
	NBModel7.trainModel()
	#NBModel7.printModel()
	model7res = NBModel7.testModel()

	NBModel8 = Model("8")
	NBModel8.Adults = NBModel.Adults
	NBModel8.partitionAdults(8)
	NBModel8.trainModel()
	#NBModel8.printModel()
	model8res = NBModel8.testModel()

	NBModel9 = Model("9")
	NBModel9.Adults = NBModel.Adults
	NBModel9.partitionAdults(9)
	NBModel9.trainModel()
	#NBModel9.printModel()
	model9res = NBModel9.testModel()

	NBModel10 = Model("10")
	NBModel10.Adults = NBModel.Adults
	NBModel10.partitionAdults(10)
	NBModel10.trainModel()
	#NBModel10.printModel()
	model10res = NBModel10.testModel()

	'''
	AVERAGES CALCULATIONS
	'''
	averageMicroP = model1res[0] + model2res[0] + model3res[0] + model4res[0] + model5res[0] + model6res[0] + model7res[0] + model8res[0] + model9res[0] + model10res[0]
	averageMicroP = float(averageMicroP / 10)

	averageMicroS = model1res[1] + model2res[1] + model3res[1] + model4res[1] + model5res[1] + model6res[1] + model7res[1] + model8res[1] + model9res[1] + model10res[1]
	averageMicroS = float(averageMicroS / 10)

	averageMicroF = model1res[2] + model2res[2] + model3res[2] + model4res[2] + model5res[2] + model6res[2] + model7res[2] + model8res[2] + model9res[2] + model10res[2]
	averageMicroF = float(averageMicroF / 10)

	averageMacroP = model1res[3] + model2res[3] + model3res[3] + model4res[3] + model5res[3] + model6res[3] + model7res[3] + model8res[3] + model9res[3] + model10res[3]
	averageMacroP = float(averageMacroP / 10)

	averageMacroS = model1res[4] + model2res[4] + model3res[4] + model4res[4] + model5res[4] + model6res[4] + model7res[4] + model8res[4] + model9res[4] + model10res[4]
	averageMacroS = float(averageMacroS / 10)

	averageMacroF = model1res[5] + model2res[5] + model3res[5] + model4res[5] + model5res[5] + model6res[5] + model7res[5] + model8res[5] + model9res[5] + model10res[5]
	averageMacroF = float(averageMacroF / 10)

	averageA = model1res[6] + model2res[6] + model3res[6] + model4res[6] + model5res[6] + model6res[6] + model7res[6] + model8res[6] + model9res[6] + model10res[6]
	averageA = float(averageA / 10)

	averagesOfResults = [averageMicroP, averageMicroS, averageMicroF, averageMacroP, averageMacroS, averageMacroF, averageA]

	writeResults(model1res, model2res, model3res, model4res, model5res, model6res, model7res, model8res, model9res, model10res, averagesOfResults)

	STOP_TIME = timeit.default_timer()
	print (str(STOP_TIME - START_TIME) + " seconds have passed")

