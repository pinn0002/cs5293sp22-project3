Name: Divya Sai Pinnamaneni

Run Procedure:

Install all the below specified libraries using the following command pipenv install modulename.

Install pipenv using pip install pipenv

Run the project using pipenv run python unredactor.py unredacted.tsv where unredactor.py is my python script and unredacted.tsv is name of input file

Run testfiles using pipenv run python -m pytest

Expected bugs

Any other file format except .tsv will not be accepted as input file format.

While calculating precision score one zerodivision warning is created.

For some reason, process is getting killed when I run in GCP but through pycharm I'm able to obtain output in my local machine. I have provided sample output below for confirmation.

I haven't written a function to obain latest changes from github instead I have taken all latest records to my unredactor.tsv


Libraries used in this project:

glob, sys, nltk, sklearn, pandas, copy, re, numpy, timeblob, csv, textblob


Assumptions:

For performing this project, below are my assumptions
1. As I already have training, testing, validation data separately so to perform unredaction I believed training data to be used for training machine with the corresponding names.
2. Testing sentences are used for prediction of redacted word 
3. So, I have assumed last column from unredacted.tsv need to be trained first with the corresponding names from the previous column.
4. All training sentences in unredacted.tsv are used for training and testing sentences are used for calculating predictions.
5. Output of redacted names to be displayed along with precision accuracy and f-1scores need to be displayed directly to the terminal.


Project Objective:
Main aim of this project is to predict the name which is redacted using blocks in the sentences. To make machine understand the redacted words, I'm required to train machine with the sentences containing redacted words and corresponding output name words for redacted word. After training the model, provide sentence and predict expected redacted word.

unredacted.tsv is the input file which is separated with tab space between columns.
Output of the redacted sentences is displayed to the terminal in the form of list.

Dataset used: unredacted.tsv is used which is collection of collobarative sentences from every member of this course.

Functions:

doextraction(sys.argv[-1]):

This function takes the file input from sys.argv[-1] which is nothing but the file used in run statement. For example,in  pipenv run python unredactor.py unredacted.tsv. unredacted.tsv is the position for sys.argv[-1]. Expected output from this function should return related xtrain and ytrain variables for fitting the model into.

This function reads unredacted.tsv file using pd.read_csv as a dataframe and considers only columns with names and sentences with redacted text but only considers sentences if they are training related.
names column is created as list of names for sending it as ytrain data. sentences with redacted text are sent to get_entity function for retrieving features of each sentence.And result from all features are appended to a lsit for considering it to be xtrain data.

get_entity(text):

This function takes sentence with redacted text as input and extract features of this sentence. I have used following features for detecting a sentence 
Features used:
Length of sentence
Count of spaces in a sentence

Number of words in a sentence

One gram, Two gram, Three grams length of sentences are used. 

For obtaing n-grams I have used a function wordgrams(text,n) which takes sentences and number of grams required as arguments and provide list of grams created as output and I have obtained this functionality using ngrams package available from nltk.util

Sentiment score of a sentence - For getting sentiment score I have used TextBlob function from textblob package  and TextBlob(text).sentiment.polarity is used for obtaining sentiment score of a sentence.

Redacted words - Use a regex pattern to find redacted words in a sentence and count number of those redacted words. If there are no redacted words by default consider as '0'.

length of a  sentence with no space - Find length of sentence by removing spaces in between

All these features are added to a dictionary and copied to another dictionary for not missing any of the features of sentences.

This function would return dictionary of all above features

doextractredaction(sys.argv[-1]):

This function would also take unredacted.tsv as input and provide Xtest and Ytest as output. Similar to the doextraction it reads unredacted.tsv file using pd.read_csv as a dataframe and considers only columns with names and sentences with redacted text but only consider sentences if they are testing related.
names column is created as list of names for sending it as ytest data. sentences with redacted text are sent to get_redactfeature function for retrieving features of each sentence. And result from all features are appended to a lsit for considering it to be xtest data. 

ytest values are actual result of predicted value.
xtest values are features of testing sentences.

get_redactfeatures(text):

This function takes sentence with redacted text as input and extract features of this sentence. I have used following features for detecting a sentence 
Features used:
Length of sentence
Count of spaces in a sentence

Number of words in a sentence

One gram, Two gram, Three grams length of sentences are used. 

For obtaing n-grams I have used a function wordgrams(text,n) which takes sentences and number of grams required as arguments and provide list of grams created as output and I have obtained this functionality using ngrams package available from nltk.util

Sentiment score of a sentence - For getting sentiment score I have used TextBlob function from textblob package  and TextBlob(text).sentiment.polarity is used for obtaining sentiment score of a sentence.

Redacted words - Use a regex pattern to find redacted words in a sentence and count number of those redacted words. If there are no redacted words by default consider as '0'.

length of a  sentence with no space - Find length of sentence by removing spaces in between

All these features are added to a dictionary and copied to another dictionary for not missing any of the features of sentences.

This function would return dictionary of all above features.

Use DictVectorizer to fit_transform of xtrain and xtest data for informing machine about the features in numerical formats. 

Use RandomforestClassifier with n_estimators = 100 to fit the model with xtrain and ytrain data and obtain prediction for xtest.

For obtaining, F-1 score precision and recall use functions from sklearn.metrics and pass arguments as actual values and predicted values.

Though different classifiers like LogisticRegression, KNeighboursclassifier but used RandomforestClassifier as they give better results.

Testcases:

def test_doextraction():
    checks original doextraction function returns list or not.
def test_entity():
    checks original get_entity function and returns dictionary or not.
def test_extraction():
    checks original get_redactfeatures and returns dictionary or not.    
def test_entityfeatures():
    checks original doextractredaction and check if return is empty or not.
    
    
  Sample Output:
  ![image](https://user-images.githubusercontent.com/98125050/167465555-50e4d33a-0bb8-44a0-9601-a1724174d572.png)
PS F:\Divya\cs5293sp22-project3> pipenv run python unredactor.py .\unredacted.tsv
Courtesy Notice: Pipenv found itself running within a virtual environment, so it will automatically use that environment, instead of creating its own for any project.
 You can set PIPENV_IGNORE_VIRTUALENVS=1 to force pipenv to ignore that environment and create its own instead. You can set PIPENV_VERBOSITY=-1 to suppress this warni
ng.
training done
Actual values ['Kevin Costner', 'Costner', 'Leonard Maltin', 'Bob McKimson', 'Maria Von Trapp', 'Julie', 'Julie Andrews', 'Rosie Perez', 'Merkerson', 'Ruben Santiago 
Hudson', 'Das Boot', 'Steven Dorff', 'Michael Ballhaus', 'Jean-Marie Lamour', 'Shane Black', 'Colin Firth', 'Christine Scott Thomas', 'Gaspar Noe', 'Phillipe Nahon', 
'Lisa Krueger', 'Randolph Scott', 'Fritz Lang', 'Dean Jagger', 'Clive Barker', 'Danny Elfman', 'Doug Bradley', 'David Cronenberg', "Vincent D'Onofrio", 'Shemp', 'Curl
y', 'Riff Randell', 'Robert Downey Jr', 'De Sica', 'Harry Cohn', 'Pierce Brosnon', 'Jeffrey Goines', 'James Bond', 'Jeremy Northam', 'Dil Wale Dulhaniya Le Jayenge', 
'Red Queen', 'Rock Hudson', 'Eric Stoltz', 'Daniel Greystone', 'Buzz Aldrin', 'Buzz Aldrin', 'Bela Lugosi', 'Patricia Hunter', 'Monogram', 'Bela Lugosi', 'Lois Lane',
 'Bush', 'Darwin', 'Dr. Dre', 'Hirsch', 'Queen Bee', 'Rik Mayall', 'Rik Mayall', 'Rudy Ray Moore', 'Spalding Gray', 'Willie Green', 'Scully', 'Monroe', 'Crowe', 'Zize
k', 'Dalton', 'Keane', 'Heather Grahm', 'Mitchell', 'Mermaid', 'Lew Ayres', 'Costner', 'Kutcher', 'Costner', 'Coast Guard', 'Hudson', 'Kevin Costner', 'David Bryce', 
'Wonder Showzen', 'Wonder Showzen', 'Wonder Showzen', 'Liv Ullmann', 'Sir John Gielgud', 'Bobby Van', 'Liv Ullmann', 'Lost Horizon', 'Richard Greico', 'Keith Richards
', 'Gary Busey', 'Gary Busey', 'Emanuele Crialese', 'Carrie White', 'Sadako', 'Sadako Yamamura', 'Sadako', 'Sadako', 'Hiroshi Takahashi', 'Norio Tsuruta', 'Sadako', '
Tsuruta', 'Shoko Miyaji', 'Melissa', 'Clancy B. Grass', 'Marine Corp', 'Jack', 'Full Metal Jacket', 'Ricci', 'Tarantino', 'Kill Bill', 'Amitabh Bachchan', 'Linda Cart
er', 'Mark', 'Lai', "Jalal Merhi's", 'Lugosi', 'Paul Gross', 'Arthur MacArthur', 'Wilma', 'Temuera', 'Andrew Gurland', 'Gerardo', 'James', 'Jefferson', 'John Wood', '
William Hurt', 'Angela Lansbury', 'Sam Jaffe', 'Tom Cruise', 'Bill Sykes', 'Nancy', 'George Fenton', 'Paul Winfield', 'Dudelson', 'Catherine', 'Daniel', 'Patrick', 'H
annah', 'Baywatch', 'Nick', 'Ryan', 'Disney', 'Hitchcock', 'Safran Foer', 'James Frain', 'Ian Hart', 'Mervyn Leroy', 'Henry Thomas', 'Jean Simmons', 'Capone', 'Julia 
Stiles', 'Edward Woodward', 'Michael Meyers', 'Kevin Bacon', 'Hackman', 'Phillip Seymour Hoffman', 'Dr Klaus', 'Barney', 'Mark', 'Tripper', 'Jon Bon Jovi', 'Meryl Str
eep', 'Natalie Portman', 'Eytan Fox', 'Aragorn', 'Elmore Leonard', 'Laura Gemser', 'Damon Wayans', 'Trent Haaga', 'Andy Copp', 'Susan George', 'Lance Henrikson', 'Jam
es Wood', 'Cameron Diaz', 'Lisa Niemi', 'Christopher Walken', 'Wendell Scott', 'Ben Stiller', 'Michael Ironside', 'Jackie Chan', 'Rob Bottin', 'Paul Koslo', "Philip G
roning's", 'Steven Seagal', 'John Heard', 'John Heard', 'Molly Ringwald', 'Ally Sheedy', 'Alfred Hitchcock', 'Anthony', 'Marisa Ryan', 'Christopher Lee', 'Christopher
 Lee', 'Virzi', 'Monica Bellucci', 'Sabrina Impacciatore', 'Daniel', 'Auteuil', 'Napoleon', 'Natalya', 'Super Mario', 'Mario Kart', 'Perfect Dark', 'Tantoo Cardinal',
 'Coyote Waits', 'Robert Redford', 'Hillerman', 'Adam', 'Hollywood', 'Gibson', 'Adam Beach', 'Tony Hillerman', 'Tony', 'Peter Falk', 'milly dowler', 'Ben Mezrich', 'S
chlesinger', 'Shirley', 'Bill Hicks', 'George Lucas', 'Baio', 'Erika Eleniak', 'Anton Newcombe', 'Bart Alle', 'Steve Deknight', 'Kevin Costner', 'Ben Randall', 'Willi
am H. Macy', 'Nikolai Cherkasov', 'Vasili', 'Olga', 'Kurasawa', 'Eisenstein', 'Connie Gilchrist', 'Angelina Jolie', 'Don Quixote', 'Rudy Ray Moore', 'Matthew Modine',
 'Charley Chase', 'Yukio Mishima', 'Wonder Showzen', 'Charles Darwin', 'Bryan Brown', 'John Larch', 'James Gandolfini', 'Audrey Hepburn', 'Jean Louis Tringtignat', 'S
usan Fletcher', 'Jesus Ponce', 'John Savage', 'Guy Ritchie', 'Spike Feresten', 'Mao Tse Dung', 'Barbara Payton', 'Park Chan-Wook', 'Luigi Bazzoni', 'Joe', 'Sabrina', 
'Hauer', "Mika Waltari's", 'Traffik', "John Leguizamo's", 'Charlie', 'Gilbert Gottfried', 'Adam Sandler', 'Jim Carrey', 'John Wayne', 'Danny Devito', 'Jack Nicholson'
, 'Robert DeNiro', 'Billy Crystal', 'Hugh Jackman', 'Guy Ritchie', 'Uma Thurman', 'Lon Chaney Jr', 'jack', 'David Lynch', 'Leo Gorcey', 'Mr Gastineau', 'Darwin', 'Ric
hard Kelly', 'Leni RiefenStahl', 'Simpson', 'Roslin', 'James Stewart', 'Jo McKenna', 'Doris', 'Louis Bernard', 'Daniel Gélin', 'Christopher Olsen', 'Hitchcock', 'Stew
art', 'Que Sera', 'Albert Hall', 'Davies', 'Haines', 'Haines', 'Davies', 'Max von Sydow', 'Von Trier', 'Lars von Trier', 'Tsui Hark', 'Von Trier', 'Max von Sydow', 'V
on Trier', 'Russo', 'Tarantino', 'Wood', 'Huston', 'Tarantino', 'Corey', 'Romero', 'Zucker', 'Kaas', 'Milland', 'Harald', 'Jonathan Flora', 'Dave Fleischer', 'Sister 
Mary', 'Taylor', 'Su Lizhen Chan', 'Papa Lazarou', 'Bambi', 'Meadows', 'Cecilia', 'Moretti', 'Unhinged', 'Jim Carrey', 'Lugosi', 'Reese Witherspoon', 'Richards', 'Jea
nne', 'Wilson', 'Ivana Milicervic', 'Porter', 'Yvaine', 'Lamia', 'Tristain', 'Robert DeNiro', 'Michelle Pfeiffer', 'Claire', 'Neil Gaiman', 'Robert', 'Ricky Gervais',
 'Nietzsche', 'Anne', 'Wentworth', 'Henrietta', 'Kimberly', 'Richard Adams', 'Jane Fonda', 'Michael Douglas', 'Jack Lemmon', 'David Ketchum', 'Bruce Shelly', 'Ecclest
on', 'Valley Girl', 'Claudius', 'Hearn', 'Miike', 'Diane', 'Ron Howard', 'Camilla', 'Adam', 'David Lynch', 'David Norwell', 'Adam', 'Norwell', 'Curtis', 'Rita Haywort
h', 'Rusty', 'Ryan Latshaw', 'Ed Wood', 'Stevens', 'Linnea', 'Stanley', 'Stanley', 'Stanley', 'Stanley', 'Stanley', 'Stanley', 'Jane Fonda', 'Stanley', 'Stanley', 'Ro
bert DeNiro', 'Stanley', 'Christopher', 'Walken', 'Marner', 'Christopher Freakin', 'Walken', 'Christopher Walken', 'Jason Connery', 'Christopher Walken', 'Walken', 'W
alken', 'Happy Cat', 'Christopher', 'Walken', 'Jason Connery', 'Corin', 'Puss', 'Corin', 'Emilio', 'Barbara Bouchet', 'Macy Gray', 'Julie Andrews', 'Lili Smith', 'Jul
ie Andrews', 'Blake Edwards', 'Robert McKimson', 'Hugh Grant', 'Kannathil Muthamittal', 'Geoffrey Land', 'Paul Lukas', 'Geraldine Fitzgerald', 'James Cameron', 'Park'
, 'Epatha Merkerson', 'Ray', 'Rachel', 'Merkerson', 'Rachel', 'Peter Yates', 'Albert Finney', 'Richard Brooks', 'Truman Capote', 'Conrad Hall', 'Quincy Jones', 'Georg
e', 'Joseph Gordon-Levitt', 'Milton David Jr.', 'Jack', 'Reno', 'Frodo', 'George Scott', 'Jim Varney', 'Kellie Martins', 'Jason Donovan', 'Rob Lowe', 'Mel Gibson', 'M
el Brooks', 'Juhi chawla', 'Martin Sheen', 'Elton John', 'Albert', 'Sebastian Stan', "Marc Blitzstein's", 'Bender', 'Bronson', 'Roth', 'Eva', 'Sandler', 'Azumi', 'Hel
len Sharp', 'Mr. Costner', 'Ashton Kutcher', 'David Norwell', 'Ronald Reagan', 'Sheila Bromley', 'Milos Forman', 'Kutcher', 'Rita Hayworth', 'Eleanor Bergstein', 'Alv
in Sargent', 'Jacob Goodnight', 'Ted Koppel', 'Tyra', 'Elton John', 'Catherine Deane', 'Flea', 'Elizabeth Montgomery', 'Claire Daines', 'Laura Bowman', 'Jaleel White'
, 'Bob', 'Jean Reno', 'Besson', 'Besson', 'Jean Reno', 'Serra', 'Nikita', 'Nikita', 'Nikita Taylor', 'Luc Besson', 'Kim Novak', 'Angelina Jolie', 'Stephane Rideau', '
Kevin Conroy', 'Adam Sandler', 'Don Cheadle', 'Alec Guiness', 'Henri Verneuil', 'Sam Mendes', 'Jude Law', 'Jacob Goodnight', 'Ted Koppel', 'Tyra', 'Elton John', 'Cath
erine Deane', 'Flea', 'Elizabeth Montgomery', 'Claire Daines', 'Laura Bowman', 'Jaleel White', 'Jodie', 'Detlef Sierck', 'Jacobb', 'Earth', 'Frankie Diomede', 'Jonath
an Kaplan', 'Sammo Hung', 'Jimmy Stewart', 'Amrita Rao', 'Albuquerque', 'Neal Jimenez', 'Keanu Reeves', "David Cronenberg's", 'Shakespeare', 'Renny Harlin', 'Samuel J
ackson', 'Shakespeare', 'Derek', 'Derek Jacobi', 'Emilio Estevez', 'Shara Reiner', 'Wong Long', 'Luise Rainer', 'George Takei', 'Paul Muni', 'Sidney Franklin', 'O-Lan
', 'Wang Lung', 'Tim Matheson', 'Jack Palance', 'Debbie', 'Rock', 'Sofia Coppola', 'David Hasselhoff', 'Linda Blair', 'Linda', 'Patrick Stewart', 'Carly Pope', 'Rena 
Owen', 'Adrian Pauls', 'Jackie Chan', 'Patrick Bergin', 'Nichols', 'Michael Shannon', 'David Gordon Green', 'Nichols', 'Nichols', 'Jeff Nichols', 'Turner', 'Turner', 
'Mr. Costner', 'Ashton Kutcher', 'Kevin Costner', 'Randall Will', 'Ronald Harwood', 'Mary Poppins', 'Lalo Schifrin', "Patrick O'Neal", 'Vincent', 'Eva Gabor', 'Mark S
tevens', 'Tamilyn Tomita', 'Jason Scott Lee', 'William Hurt', 'Peter Weller', 'Hardy Kruger Jr', 'Natasha McElhone', 'Michael Brandon', 'Michael Brandon', 'Susan Swif
t', 'Walken', 'Lars Von Trier', 'Villaronga', 'Texas Chainsaw Massacre', 'Fantasy', 'Katharina', 'Max Hartmann', 'Agnieszka Holland', 'Stanley Kubrick', "Vincent D'On
ofrio", 'DeNiro', 'First Mate', 'Gary Busey', 'Sadako', 'Brolin', 'Boll', 'Lebowski', 'Murry Lerner', 'Rick Marshall']
Predicted values ['Toby Stephens' 'Winfrey' 'Peter Sellers' 'Carl Foreman'
 'Alfred Hitchcock' 'Allan' 'Carlos Mencia' 'Buzz Aldrin' 'Roosevelt'
 'Nathaniel Hawthorne' 'Roseanne' 'Josh Flitter' 'Michael Rooker'
 'Joseph K. Images' 'Bob Balaban' 'Sarah Kants' 'Rosalind Russell movies'
 'Dan Duryea' 'Shelly Winters' 'William Boyd' 'Randolph Scott'
 'jay walker' 'Peter Dalle' 'George Brent' 'Jenny Latour' 'Doug Bradley'
 'Juliette Binoche' 'Robert E. Sherwood' 'Shemp' 'Marie' 'Rock Hudson'
 'Frederic March' 'Scrooge' 'Bob Barker' 'Christopher' 'Rebecca Gibney'
 'Mr. Knotts' 'Andre Braugher' 'Art Director Dave Milton' 'Schindler'
 'Rock Hudson' 'Eric Stoltz' 'Daniel Greystone' 'Buzz Aldrin'
 'Buzz Aldrin' 'Chris Cooper' 'Patricia Hunter' 'Raisouli' 'Bill Paxton'
 'Chad Lowe' 'Bush' 'Reiner' 'Stiller' 'Atwill' 'Queen Bee' 'Bill Murray'
 'Rik Mayall' 'Charles Bronson' 'Kelly Preston' 'Ciaran Hinds' 'Gundam'
 'Tambor' 'Lando' 'Mehta' 'Johnny' 'Hyeon' 'Jia Hongsheng' 'Williams'
 'Antonio' 'von Trier' 'Antonio' 'Celeste' 'Camilla' 'Natilie Portman'
 'Sadako' 'Richard Nixon' 'George Hilton' 'Edmond Rackham' 'William Hurt'
 'Barbara Stanwyck' 'Hank Landry' 'Vampiros Lesbos' 'Edie Falco'
 'Heavy Metal' 'Laura Linney' 'Walter Brennan' 'Robert De Niro'
 'Emile Zola' 'Nancy Drew' 'Tara Fitzgerald' 'Jack Arnold' 'Duguay'
 'Eddie Considine' 'Sadako' 'Burton' "Chan-wook Park's" 'Amanda Redman'
 'Nguyen' 'Winfrey' 'Steve Carell' 'Vincent' 'Sidney J. Furie'
 'Darth Vader' 'Fair' 'Martin Scorcese' 'Kathy' 'Jane Eyre' 'Tsui Hark'
 'Jessica Cauffiel' 'Louis Sachar' 'Asin' 'Rai' 'Kenny Doughty' 'Alison'
 'Wyatt Earp' 'Jessica Cauffiel' 'Libby' 'Leopold' 'Stephanie Meyer'
 'Stanley' 'Peter' 'Eva Gabor' 'Vic Mizzy' 'Yukie Nakama'
 'Sally Kellerman' 'Hitchcock' 'Luke Perry' 'Shug Avery' 'Simon'
 'Vince McMahon' 'Robert DeNiro' 'Kay Lenz' 'Tsui Hark' 'Godard' 'Matthew'
 'Altman' 'Del Toro' 'any1' 'Meh' 'Jet Li' 'Schindler' 'Safran Foer'
 'Matt Newton' 'Goldberg' 'Mike Connors' 'Willie Green' 'Adam Sandler'
 'Bernie' 'Darren Stein' 'Charlton Heston' 'Park Chan-wook' 'John Glover'
 'Pollard' 'Richard C. Sarafian' 'Lovelace' 'Sadako' 'John' 'Stanley'
 'Jon Bon Jovi' 'Mohamed Majd' 'Natalie Portman' 'Von Trier' 'Sabrina'
 'Ashton Kutcher' 'Stephen King' 'Ciaran Hinds' 'Brad Pitt' 'Verhoeven'
 'Buzz Aldrin' 'Mark Borchardt' 'Edie Falco' 'Gordon Clapp' 'Jim Carrey'
 'David Attenborough' 'Bromwell High' 'Bobby Fischer' 'Michael Ironside'
 'Craig Sheffer' 'Lane Smith' 'Jim Carrey' 'George C. Scott'
 'Dennis Hopper' 'Fargoesque' 'Adam Jones' 'Chan-wook Park' 'Jackie Chan'
 'Millard Mitchell' 'Hari Om' 'Bette Davis' 'Detective Garner'
 'Patricia Pepoire' 'Peter' 'Jake Gyllenhaal' 'Christopher Walken'
 'Bachan' 'Melissa' 'Red Hair' 'Pollard' 'Uma Thurman' 'Alan Arkin'
 'Orson Welles' 'Jeremy Clarkson' 'Victor Borge' 'Chris Sarandon'
 'Tom Brown' 'Kate' 'Tom Brown' 'Eilers' 'Errol Flynn' 'Chris Sarandon'
 'Kate' 'Peter Falk' 'Monica Keena' 'Goldie Hawn' 'Miss Monroe' 'Antoine'
 'Robin Hood' 'Steve Martin' 'Iris' 'Marlon Brando' 'David Boreanaz'
 'Tsui Hark' 'Jeff Lieberman' 'George Hilton' 'Ally McBeal'
 'Max Fleischer' 'Jeff Lieberman' 'Reiner' 'Bled' "Arnold's" 'Scott Wolf'
 'Sammy Davis Jr.' 'Charles Korvin' 'Edison Chen' "Pearl Bailey's"
 'Robert De Niro' 'Bruno Nicolai' 'Martin Sexton' 'Edmond Rackham'
 'Jeremy Northam' 'Dan Duryea' 'Mark Sandrich' 'Mélanie Thierry'
 'Charles Dickens' 'Landlord Edgar Kennedy' 'Luther Luckett'
 'Yukie Nakama' 'Miss Young' 'Peter Carey' 'Edmond Rackham' 'Jerry Orbach'
 'Melvin Douglas' 'David Boreanaz' 'Marion Davies' 'Lin' 'Ang Lee' 'Radha'
 'Hector Cordoba' 'Margera' 'Hannibal Rising' 'Scacchi'
 'Brandon Dicamillo' 'Adam Sandler' 'Chris Gunn' 'Gary Busey'
 'Meryl Streep' 'Marta Belengur' 'Anne Bancroft' 'Anne Bancroft'
 "Gail O'Grady" 'Rosie Perez' 'Uma Thurman' 'Ralph Bakshis' 'Alex'
 'Aidan Quinn' 'Neil Simon' 'Robert Evans' 'Sadako' 'Peter Ustinov'
 'Ashton Kutcher' 'Dunaway' 'Nguyen' 'Bridget Fonda' 'Bill Sikes' 'Chevy'
 'Michael Caine' 'Robert Morse' "Vincent D'Onofrio" "D'Onofrio" 'Michael'
 'François' 'Robert Morse' 'Jeremy' 'Jeremy' 'Bogart' 'Bogart'
 'Geoffrey Land' 'Hitchcock' 'Rupert Grint' 'Barrymore' 'Chipmunks'
 'Randy Edelman' 'von Trier' 'Baldi' 'Dodsworth' 'Joey' 'Davies'
 'Ed Norton' 'Davis' 'Fisher' 'Sadako' 'Bach' 'Adriana' 'Stella'
 'Michael Madsen' 'Penny Marshall' 'Alain Delon' 'Thomas'
 'Sacha Baron Cohen' 'Adam Sandler' 'Bambi' 'Rebecca' 'Kennedy' 'Stewart'
 'Jovovich' 'Jud Nelson' 'Morris' 'Quenton Tarantino' 'Kurosawa' 'Pichel'
 'George' 'David Strathairn' 'Davies' 'Carlos' 'David' 'Minnelli'
 'Edward Furlong' 'Mary Steenburgen' 'Sanjay' 'Ruth Gordon' 'Billie'
 'Lola Albright' 'Tim Curry' 'Cage' 'Eva Gabor' 'Eva Gabor' 'Waldemar'
 'Willie Nelson' 'Bob Barker' 'Franka Pontente' 'Esther Kahn'
 "Bob Ballard's" 'Monica Keena' 'Northfork' 'Salman Khan' 'Benjamin'
 'Riley' 'Damon' 'Mehta' 'Ahna Capri' 'Fernack' 'Matt' 'Shakespeare'
 'Leopold Kessler' 'Lutz' 'William' 'Huston' 'Leopold Kessler' 'Bride'
 "Sean Young's" 'Antoine' 'Hartley' 'Evelyn' 'Garrett' 'Columbo' 'Elenore'
 'Stanley' 'Stanley' 'Rob Roy' 'Omar Sharif' 'Stiller' 'De Niro'
 'Bruno Nicolai' 'Blunder' 'Lorna Doone' 'Davies' 'Bogart'
 'Caroline Dhavernas' 'Huston' 'Brokeback Mountain' 'Jason Connery'
 'Christopher Walken' 'Nelson' 'Jeanne' 'John Ford' 'Dee Pollock' 'Stella'
 'Phil Sheridan' 'Inman' 'Ajay' 'Holly' 'Sadako' 'Sybil Danning'
 'Hitchcock' 'Butch Patrick' 'LUC BESSON' 'Dennis Spooner' 'Toby Stephens'
 'Cecilia Cheung' 'Emily Posa' 'Kannathil Muthamittal' 'Geoffrey Land'
 'Paul Lukas' 'Jennifer Jason Leigh' 'Yonica Babyya' 'Jack'
 'Shirley MacLaine' 'Ben' 'Sadako' 'Jim Brown' 'Hrabal' 'Lucas Black'
 'Orlando Jones' 'George Sanders' 'Griffin Dunne' 'Sprecher'
 'Emily Watson' 'Dahmer' 'Nadine Van der Velde' 'Dewey Robinson' 'Sita'
 'Sita' 'Jason' 'Bruce Willis' 'LUC BESSON' 'Alistair Cooke'
 'Robert Morley' 'Jude Law' 'LUC BESSON' 'Jack Jones' 'John Ritter'
 'Roy Scheider' 'LUC BESSON' 'Sterno' 'Luke Skywalker'
 'Robert E. Sherwood' 'Paulie' 'Brendon' 'Boll' 'MM' 'Vincent' 'Trier'
 'Abel Ferrara' 'Bob Balaban' 'Edmond Rackham' 'Frederic Forrest'
 'Howard Hughes' 'Walter Pidgeon' 'David Palmer' 'Celeste' 'Dylan Klebold'
 'Peter Bogdonavich' 'Uncle Kessler' 'Richard Widmark' 'Vin Diesel' 'Matt'
 'Ellen Page' 'Charlton Heston' 'John' 'Sachin Dev Burman' 'Wesley Snipes'
 'Piers Morgan' 'Hector Perez' 'Tom' "John Woo's" 'Knotts' 'Arnold'
 'Lawrence' 'Foley' 'Denton' 'Sadako' 'Crispin Glover' 'Rob Reiner'
 'Artimisia' 'Chris Sarandon' 'François Coste' 'Janine Turner'
 'Sally Eilers' 'Errol Flynn' 'Robert Morse' 'Daniel Auteuil' 'Henry King'
 'Beautiful' 'Richard Widmark' 'Vin Diesel' 'Matt' 'Ellen Page'
 'Charlton Heston' 'John' 'Sachin Dev Burman' 'Wesley Snipes'
 'Piers Morgan' 'Hector Perez' 'Holly' 'Bruno Nicolai' 'Wilder' 'Vidor'
 'Irvin Kershner' 'Matthew Weiner' 'Skerritt' 'Frank Launder' 'Eric Monte'
 'Miss Monroe' 'James Cagney' 'Ethan Hawke' 'Leopold Kessler'
 'Shakespeare' 'Bill Murray' 'Burt Reynolds' 'Robert DeNiro' 'Eddie'
 'John Holmes' 'Robert DeNiro' 'Citizen Kane' 'Tsui Hark' 'Dana Andrews'
 'Robin Bailey' 'Von Trier' 'Richard Shepard' 'Danny' 'Billy Bob'
 'Jackie Chan' 'Claude Rains' 'Brenda' 'Lars' 'Lloyd Kaufman'
precison1: 0.030782438067206274
F:\Divya\cs5293sp22-project3\venv\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in lab
els with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
recall: 0.03361540348295315





