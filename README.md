# D&D_ML_CR_Generator
Here is a machine learning algorithm that was trained on 700+ DND(5e) monsters.

The program predicts CR using the 8 other statistics of the monster
AC, STR, DEX, INT, WIS, CON, CHA, and HP

When playing D&D people use custom monsters sometimes and this does not come with the challenge rating the other monsters come with when published in official literature.
(A challenge rating (CR) is the average level a party of four players should be to have a medium to easy time defeating the monster)

By filling out the monsters_query.csv file with the stats of your custom monster it will predict appropriate CRs for each line and give you the accuracy score it received on its  training

1. Download zip folder
2. Fill out monsters_query.csv
3. Open Monster_Lin_Reg.py in a Python IDE (Spyder, for example)
4. Keep or comment out reference graphs
5. Execute and enjoy your CRs
