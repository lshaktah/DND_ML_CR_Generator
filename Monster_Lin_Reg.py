import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('monsters_training.csv')
print(df.head())

# Step 1, understand your data
# There are 8 statistics that go into making a Challenge Rating(CR) for a monster in DND
# AC: Armor Class
# STR: Strength
# DEX: Dexterity
# CON: Constitution
# INT: Intelligence
# WIS: Wisdom
# CHA: Charisma
# HP: Hit points
# The CR is the average level a group of 4 should be to defeat the monster
# When people make their own monsters it does not come with a CR
# So to help those that want to gauge their own creations to the core monsters published with DND they can use this

plt.scatter(df['AC'], df['CR'], marker=".", color='red')  # lets us see independent variables vs dependent
plt.ylabel("CR")
plt.xlabel("AC")
plt.show()


plt.scatter(df['STR'], df['CR'], marker=".", color='red')  # lets us see independent variables vs dependent
plt.ylabel("CR")
plt.xlabel("STR")
plt.show()

plt.scatter(df['DEX'], df['CR'], marker=".", color='red')  # lets us see independent variables vs dependent
plt.ylabel("CR")
plt.xlabel("DEX")
plt.show()

plt.scatter(df['CON'], df['CR'], marker=".", color='red')  # lets us see independent variables vs dependent
plt.ylabel("CR")
plt.xlabel("CON")
plt.show()

plt.scatter(df['INT'], df['CR'], marker=".", color='red')  # lets us see independent variables vs dependent
plt.ylabel("CR")
plt.xlabel("INT")
plt.show()

plt.scatter(df['WIS'], df['CR'], marker=".", color='red')  # lets us see independent variables vs dependent
plt.ylabel("CR")
plt.xlabel("WIS")
plt.show()

plt.scatter(df['CHA'], df['CR'], marker=".", color='red')  # lets us see independent variables vs dependent
plt.ylabel("CR")
plt.xlabel("CHA")
plt.show()

plt.scatter(df['HP'], df['CR'], marker=".", color='red')  # lets us see independent variables vs dependent
plt.ylabel("CR")
plt.xlabel("HP")
plt.show()

# Step 2: Drop un-needed data and handle Missing Data
# This has no un-needed data
# if you don't have a lot missing
length_before = len(df)
df.dropna()  # will remove rows with any missing data
length_after = len(df)
print(length_before - length_after, "rows were dropped")

# Step 3: Define X and Y
X = df.drop(labels=['CR'], axis=1)

Y = df.CR.values
Y = Y.astype('int')

# Step 4: Split data into test and training data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)
# print("X_train:\n", X_train, "\nX_Test:", X_test, "\nY_train:", Y_train, "\nY_Test:", Y_test)


# Step 5: Define the Model
from sklearn import linear_model

reg = linear_model.LinearRegression()  # This just created the model and now it can be used

# so now fit it to our X and Y training sets
reg.fit(df[['AC', 'STR', 'DEX', 'CON', 'INT', 'WIS', 'CHA', 'HP']], df.CR)

# Step 6: Test it

acc = reg.score(X_test, Y_test)  # This gives us the final accuracy of the model
print("Accuracy Score:", acc * 100, "%")

# This prints the weights of the various contributing features on training CR prediction
weights = pd.Series(reg.coef_, X.columns.values)
print(weights)

# Lastly, we need to be able to feed it a monster and have it guess it
# Enter your queries in the monsters_query.csv and it will predict it
# Use the accuracy score to determine if this is acceptable for your custom monsters
X_functional = pd.read_csv('monsters_query.csv')
query = reg.predict(X_functional)

i = 1
for CR in query:
    print("CR Query", i, ":", CR)
    i += 1

