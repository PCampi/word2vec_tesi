# coding: utf-8

"""List of stopwords."""
from nltk.corpus import stopwords


_prepositions = {'anch', 'pi', 'pu', 'lì', 'là', 'di', 'a', 'da', 'in', 'con',
                 'selfu', 'per', 'tra', 'fra', 'perchè'}
_months = {'Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno',
           'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre',
           'Dicembre'}
_names = {'Hazel', 'Gus', 'Walden', 'Brahma',
          'enchiladas', 'Isaac', 'Patrick', 'Margo', 'Roth',
          'Spiegelman', 'Ben', 'Starling', 'Mizuki', 'Sakaki',
          'Ando', 'Miles', 'Marie', 'Alaska', 'Buck', 'Mulligan',
          'Stephen', 'Dedalus', 'Raskòlnikov', 'Lizavèta', 'Alena',
          'Ivànovna', 'Emilio', 'Brentani', 'Angiolina', 'Sorniani',
          'Merighi', 'Zeno', 'Phoebe', 'Hal', 'Holden', 'Petròvič',
          'Goljàdkin', 'Karolina', 'Krestjàn', 'Ivànovič', 'Karamazov',
          'Dmitrij', 'Alesa', 'Smerdjakov', 'Fedor', 'Pavlovič', 'Grusenka',
          'Antolini', 'Tina', 'Janine', 'Reymont', 'Leonora', 'Christine',
          'Ingrid', 'Lindgren', 'Telander', 'Lyra', 'daimon', 'Pantalaimon',
          'Pan', 'Will', 'Serafina', 'Pekkala', 'Lord', 'Asriel', 'Coulter',
          'aletiometro', 'Ingoiatori', 'gyziani', 'Faa', 'John', 'polvere',
          'gyziana', 'gyziano', 'gyziane', 'Iorek', 'Byrnison', 'Farder',
          'Coram', 'Lee', 'Scoresby', 'Bolvangar', 'Magisterium', 'Roger',
          'Iofur', 'Cittagazze', 'Parry', 'Balthamos', 'Baruch', 'Metatron',
          'Boreal', 'Kathy', 'Tommy', 'Hailsham', 'Ruth', 'Kath', 'Cottages',
          'Madame', 'Keffers', 'Miss', 'Emily', 'Sumire', 'Myu', 'Perboni',
          'Baretti', 'Robetti', 'Derossi', 'Votini', 'Nelli', 'Stardi',
          'Precossi', 'Tom', 'Sawyer', 'Huck', 'Thatcher', 'Polly', 'Sid',
          'Csikszentmihalyi', 'Enfield', 'Tennis', 'Academy', 'ETA',
          'Mario', 'Avril', 'Freer', 'Pemulis', 'Orin', 'Michael', 'Stice',
          'Wayne', 'Possalthwaite', 'Demerol', 'Don', 'Gately', 'Kate',
          'Pat', 'Joelle', 'van', 'Dyne', 'Gompert', 'Montesian', 'Ken',
          'Randy', 'Bruce', 'Tiny', 'Doony', 'Geoffrey', 'Day', 'Antitoi',
          'Lucien', 'Bertrand', 'Chillingworth', 'Roger', 'Hester', 'Pearl',
          'Dimmesdale', 'Northanger', 'Emma', 'Darcy', 'Jane', 'Catherine',
          'Morland', 'Sally', 'Elinor', 'Willoughby', 'Sir', 'Dashwood',
          'Marianne', 'Elizabeth', 'Bennet', 'Ferrars', 'Mansfield', 'Park',
          'Fanny', 'Price', 'Highbury', 'Anne', 'Clay', 'Mr.', 'Mr', 'Mrs.',
          'Mrs', 'Harville', 'Nikolàj', 'Vedenjapin', 'Nikolàevich',
          'Jura', 'Zivago', 'Ivàn', 'Ivànovich', 'Voskobòjnikov',
          'Kologrivov', 'Pavel', 'Dupljanka', 'Màrija', 'Misha', 'Gordon',
          'Grigorij', 'Osìpovich', 'Tiverzin', 'Vedenjapin', 'Nika',
          'Olja', 'Demin', 'Amàlija', 'Kàrlovna', 'Faina', 'Fetisov',
          'Antipov', 'Tiverzin', 'Savelij', 'Nikitiè', 'Jusupka',
          'Gimazetdin', 'Chudoleev', 'Sòkolov', 'Patulja', 'Marfa',
          'Gavrìlovna', 'Sventickij', 'Komaròv', 'Dudorov',
          'Marco', 'Stanley', 'Fogg', 'Kitty', 'Wu', 'Victor', 'Effing',
          'Oklahoma', 'Joad', 'Texas', 'Panhandle', 'Mexico', 'New',
          'Route', 'Rosa', 'Tea', 'Connie', 'Noè', 'Winfield', 'Cassy',
          'Arizona', 'Casy', 'California', 'Ramsay', 'Skye', 'Augustus',
          'Carmichael', 'Paul', 'Rayley', 'Minta', 'Doyle', 'Prue',
          'Andrew', 'Camilla', 'James', 'Lily', 'Briscoe', 'Alchimista',
          'Santiago', 'Melchisedek', 'Andalusia', 'Tarifa', 'piramidi',
          'Urim', 'Tumim', 'Fatima', 'Piedra'}

_months = _months.union({m.lower() for m in _months})
_names = _names.union({n.lower() for n in _names})
_tmp = _prepositions.union(_months)
_tmp = _tmp.union(_names)
_tmp = _tmp.union(stopwords.words('italian'))

my_stopwords = _tmp.copy()
