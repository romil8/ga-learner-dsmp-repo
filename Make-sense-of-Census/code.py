# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
data= np.genfromtxt(path, delimiter=",", skip_header=1)
print("\nData: \n\n", data)

print("\nType of data: \n\n", type(data))

census= np.concatenate((new_record , data), axis=0)
print(census)





# --------------
#Code starts here
import numpy as np
age= np.array(census[:,0])

max_age= np.amax(age)
print(max_age)

min_age= np.amin(age)
print(min_age)

age_mean= np.mean(age)
print(age_mean)

age_std= np.std(age)
print(age_std)


# --------------
#Code starts here
races= census[:, 2].astype('int')
race_0= census[races == 0].astype('int')
race_1= census[races == 1].astype('int')
race_2= census[races == 2].astype('int')
race_3= census[races == 3].astype('int')
race_4= census[races == 4].astype('int')
print(race_0)
len_0= len(race_0)
len_1= len(race_1)
len_2= len(race_2)
len_3= len(race_3)
len_4= len(race_4)
lens= [len_0, len_1, len_2, len_3, len_4]
print(lens)
minority_race= lens.index(min(lens))


# --------------
#Code starts here
senior_citizens= census[census[:,0]>60]
working_hours_sum= senior_citizens.sum(axis=0)[6]
print(working_hours_sum)

senior_citizens_len= len(senior_citizens)
print(senior_citizens_len)

avg_working_hours= working_hours_sum / senior_citizens_len
print(avg_working_hours)


# --------------
#Code starts here
high= census[census[:,1]>10]
low= census[census[:,1]<=10]
avg_pay_high= high[:,7].mean()
print(avg_pay_high)
low=census[census[:,1]<=10]

#Finding the average pay
avg_pay_low=low[:,7].mean()

#Printing the average pay
print(avg_pay_low)


