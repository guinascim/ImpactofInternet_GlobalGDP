-- MySQL dump 10.13  Distrib 8.0.29, for Win64 (x86_64)
--
-- Host: localhost    Database: nuestro_amazon
-- ------------------------------------------------------
-- Server version	8.0.29

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `categories`
--

DROP TABLE IF EXISTS `categories`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `categories` (
  `CategoryID` int NOT NULL AUTO_INCREMENT,
  `CategoryName` varchar(20) DEFAULT NULL,
  `DescriptionText` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`CategoryID`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `categories`
--

LOCK TABLES `categories` WRITE;
/*!40000 ALTER TABLE `categories` DISABLE KEYS */;
INSERT INTO `categories` VALUES (1,'Beverages','Soft drinkscoffees, teas, beers, and ales'),(2,'Condiments','Sweet and savory sauces, relishes, spreads, and seasonings'),(3,'Confections','Desserts, candies, and sweet breads'),(4,'Dairy Products','Cheeses'),(5,'Grains/Cereals','Breads, crackers, pasta, and cereal'),(6,'Meat/Poultry','Prepared meats'),(7,'Produce','Dried fruit and bean curd'),(8,'Seafood','Seaweed and fish');
/*!40000 ALTER TABLE `categories` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `customers`
--

DROP TABLE IF EXISTS `customers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `customers` (
  `CustomerID` int NOT NULL AUTO_INCREMENT,
  `CustomerName` varchar(50) DEFAULT NULL,
  `ContractName` varchar(20) DEFAULT NULL,
  `Address` varchar(80) DEFAULT NULL,
  `City` varchar(20) DEFAULT NULL,
  `PostalCode` varchar(10) DEFAULT NULL,
  `Country` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`CustomerID`)
) ENGINE=InnoDB AUTO_INCREMENT=92 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `customers`
--

LOCK TABLES `customers` WRITE;
/*!40000 ALTER TABLE `customers` DISABLE KEYS */;
INSERT INTO `customers` VALUES (1,'Alfreds Futterkiste','Maria Anders','Obere Str. 57','Berlin','12209','Germany'),(2,'Ana Trujillo Emparedados y helados','Ana Trujillo','Avda. de la Constitución 2222','México D.F.','05021','Mexico'),(3,'Antonio Moreno Taquería','Antonio Moreno','Mataderos 2312','México D.F.','05023','Mexico'),(4,'Around the Horn','Thomas Hardy','120 Hanover Sq.','London','WA1 1DP','UK'),(5,'Berglunds snabbköp','Christina Berglund','Berguvsvägen 8','Luleå','S-958 22','Sweden'),(6,'Blauer See Delikatessen','Hanna Moos','Forsterstr. 57','Mannheim','68306','Germany'),(7,'Blondel père et fils','Frédérique Citeaux','24 place Kléber','Strasbourg','67000','France'),(8,'Bólido Comidas preparadas','Martín Sommer','C/ Araquil 67','Madrid','28023','Spain'),(9,'Bon app','Laurence Lebihans','12 rue des Bouchers','Marseille','13008','France'),(10,'Bottom-Dollar Marketse','Elizabeth Lincoln','23 Tsawassen Blvd.','Tsawassen','T2F 8M4','Canada'),(11,'Bs Beverages','Victoria Ashworth','Fauntleroy Circus','London','EC2 5NT','UK'),(12,'Cactus Comidas para llevar','Patricio Simpson','Cerrito 333','Buenos Aires','1010','Argentina'),(13,'Centro comercial Moctezuma','Francisco Chang','Sierras de Granada 9993','México D.F.','05022','Mexico'),(14,'Chop-suey Chinese','Yang Wang','Hauptstr. 29','Bern','3012','Switzerland'),(15,'Comércio Mineiro','Pedro Afonso','Av. dos Lusíadas 23','São Paulo','05432-043','Brazil'),(16,'Consolidated Holdings','Elizabeth Brown','Berkeley Gardens 12 Brewery','London','WX1 6LT','UK'),(17,'Drachenblut Delikatessend','Sven Ottlieb','Walserweg 21','Aachen','52066','Germany'),(18,'Du monde entier','Janine Labrune','67 rue des Cinquante Otages','Nantes','44000','France'),(19,'Eastern Connection','Ann Devon','35 King George','London','WX3 6FW','UK'),(20,'Ernst Handel','Roland Mendel','Kirchgasse 6','Graz','8010','Austria'),(21,'Familia Arquibaldo','Aria Cruz','Rua Orós 92','São Paulo','05442-030','Brazil'),(22,'FISSA Fabrica Inter. Salchichas S.A.','Diego Roel','C/ Moralzarzal 86','Madrid','28034','Spain'),(23,'Folies gourmandes','Martine Rancé','184 chaussée de Tournai','Lille','59000','France'),(24,'Folk och fä HB','Maria Larsson','Åkergatan 24','Bräcke','S-844 67','Sweden'),(25,'Frankenversand','Peter Franken','Berliner Platz 43','München','80805','Germany'),(26,'France restauration','Carine Schmitt','54 rue Royale','Nantes','44000','France'),(27,'Franchi S.p.A.','Paolo Accorti','Via Monte Bianco 34','Torino','10100','Italy'),(28,'Furia Bacalhau e Frutos do Mar','Lino Rodriguez','Jardim das rosas n. 32','Lisboa','1675','Portugal'),(29,'Galería del gastrónomo','Eduardo Saavedra','Rambla de Cataluña 23','Barcelona','08022','Spain'),(30,'Godos Cocina Típica','José Pedro Freyre','C/ Romero 33','Sevilla','41101','Spain'),(31,'Gourmet Lanchonetes','André Fonseca','Av. Brasil 442','Campinas','04876-786','Brazil'),(32,'Great Lakes Food Market','Howard Snyder','2732 Baker Blvd.','Eugene','97403','USA'),(33,'GROSELLA-Restaurante','Manuel Pereira','5ª Ave. Los Palos Grandes','Caracas','1081','Venezuela'),(34,'Hanari Carnes','Mario Pontes','Rua do Paço 67','Rio de Janeiro','05454-876','Brazil'),(35,'HILARIÓN-Abastos','Carlos Hernández','Carrera 22 con Ave. Carlos Soublette #8-35','San Cristóbal','5022','Venezuela'),(36,'Hungry Coyote Import Store','Yoshi Latimer','City Center Plaza 516 Main St.','Elgin','97827','USA'),(37,'Hungry Owl All-Night Grocers','Patricia McKenna','8 Johnstown Road','Cork','','Ireland'),(38,'Island Trading','Helen Bennett','Garden House Crowther Way','Cowes','PO31 7PJ','UK'),(39,'Königlich Essen','Philip Cramer','Maubelstr. 90','Brandenburg','14776','Germany'),(40,'La corne dabondance','Daniel Tonini','67 avenue de l Europe','Versailles','78000','France'),(41,'La maison dAsie','Annette Roulet','1 rue Alsace-Lorraine','Toulouse','31000','France'),(42,'Laughing Bacchus Wine Cellars','Yoshi Tannamuri','1900 Oak St.','Vancouver','V3F 2K1','Canada'),(43,'Lazy K Kountry Store','John Steel','12 Orchestra Terrace','Walla Walla','99362','USA'),(44,'Lehmanns Marktstand','Renate Messner','Magazinweg 7','Frankfurt a.M.','60528','Germany'),(45,'Lets Stop N Shop','Jaime Yorres','87 Polk St. Suite 5','San Francisco','94117','USA'),(46,'LILA-Supermercado','Carlos González','Carrera 52 con Ave. Bolívar #65-98 Llano Largo','Barquisimeto','3508','Venezuela'),(47,'LINO-Delicateses','Felipe Izquierdo','Ave. 5 de Mayo Porlamar','I. de Margarita','4980','Venezuela'),(48,'Lonesome Pine Restaurant','Fran Wilson','89 Chiaroscuro Rd.','Portland','97219','USA'),(49,'Magazzini Alimentari Riuniti','Giovanni Rovelli','Via Ludovico il Moro 22','Bergamo','24100','Italy'),(50,'Maison Dewey','Catherine Dewey','Rue Joseph-Bens 532','Bruxelles','B-1180','Belgium'),(51,'Mère Paillarde','Jean Fresnière','43 rue St. Laurent','Montréal','H1J 1C3','Canada'),(52,'Morgenstern Gesundkost','Alexander Feuer','Heerstr. 22','Leipzig','04179','Germany'),(53,'North/South','Simon Crowther','South House 300 Queensbridge','London','SW7 1RZ','UK'),(54,'Océano Atlántico Ltda.','Yvonne Moncada','Ing. Gustavo Moncada 8585 Piso 20-A','Buenos Aires','1010','Argentina'),(55,'Old World Delicatessen','Rene Phillips','2743 Bering St.','Anchorage','99508','USA'),(56,'Ottilies Käseladen','Henriette Pfalzheim','Mehrheimerstr. 369','Köln','50739','Germany'),(57,'Paris spécialités','Marie Bertrand','265 boulevard Charonne','Paris','75012','France'),(58,'Pericles Comidas clásicas','Guillermo Fernández','Calle Dr. Jorge Cash 321','México D.F.','05033','Mexico'),(59,'Piccolo und mehr','Georg Pipps','Geislweg 14','Salzburg','5020','Austria'),(60,'Princesa Isabel Vinhoss','Isabel de Castro','Estrada da saúde n. 58','Lisboa','1756','Portugal'),(61,'Que Delícia','Bernardo Batista','Rua da Panificadora 12','Rio de Janeiro','02389-673','Brazil'),(62,'Queen Cozinha','Lúcia Carvalho','Alameda dos Canàrios 891','São Paulo','05487-020','Brazil'),(63,'QUICK-Stop','Horst Kloss','Taucherstraße 10','Cunewalde','01307','Germany'),(64,'Rancho grande','Sergio Gutiérrez','Av. del Libertador 900','Buenos Aires','1010','Argentina'),(65,'Rattlesnake Canyon Grocery','Paula Wilson','2817 Milton Dr.','Albuquerque','87110','USA'),(66,'Reggiani Caseifici','Maurizio Moroni','Strada Provinciale 124','Reggio Emilia','42100','Italy'),(67,'Ricardo Adocicados','Janete Limeira','Av. Copacabana 267','Rio de Janeiro','02389-890','Brazil'),(68,'Richter Supermarkt','Michael Holz','Grenzacherweg 237','Genève','1203','Switzerland'),(69,'Romero y tomillo','Alejandra Camino','Gran Vía 1','Madrid','28001','Spain'),(70,'Santé Gourmet','Jonas Bergulfsen','Erling Skakkes gate 78','Stavern','4110','Norway'),(71,'Save-a-lot Markets','Jose Pavarotti','187 Suffolk Ln.','Boise','83720','USA'),(72,'Seven Seas Imports','Hari Kumar','90 Wadhurst Rd.','London','OX15 4NB','UK'),(73,'Simons bistro','Jytte Petersen','Vinbæltet 34','København','1734','Denmark'),(74,'Spécialités du monde','Dominique Perrier','25 rue Lauriston','Paris','75016','France'),(75,'Split Rail Beer & Ale','Art Braunschweiger','P.O. Box 555','Lander','82520','USA'),(76,'Suprêmes délices','Pascale Cartrain','Boulevard Tirou 255','Charleroi','B-6000','Belgium'),(77,'The Big Cheese','Liz Nixon','89 Jefferson Way Suite 2','Portland','97201','USA'),(78,'The Cracker Box','Liu Wong','55 Grizzly Peak Rd.','Butte','59801','USA'),(79,'Toms Spezialitäten','Karin Josephs','Luisenstr. 48','Münster','44087','Germany'),(80,'Tortuga Restaurante','Miguel Angel Paolino','Avda. Azteca 123','México D.F.','05033','Mexico'),(81,'Tradição Hipermercados','Anabela Domingues','Av. Inês de Castro 414','São Paulo','05634-030','Brazil'),(82,'Trails Head Gourmet Provisioners','Helvetius Nagy','722 DaVinci Blvd.','Kirkland','98034','USA'),(83,'Vaffeljernet','Palle Ibsen','Smagsløget 45','Århus','8200','Denmark'),(84,'Victuailles en stock','Mary Saveley','2 rue du Commerce','Lyon','69004','France'),(85,'Vins et alcools Chevalier','Paul Henriot','59 rue de lAbbaye','Reims','51100','France'),(86,'Die Wandernde Kuh','Rita Müller','Adenauerallee 900','Stuttgart','70563','Germany'),(87,'Wartian Herkku','Pirkko Koskitalo','Torikatu 38','Oulu','90110','Finland'),(88,'Wellington Importadora','Paula Parente','Rua do Mercado 12','Resende','08737-363','Brazil'),(89,'White Clover Markets','Karl Jablonski','305 - 14th Ave. S. Suite 3B','Seattle','98128','USA'),(90,'Wilman','Matti Karttunen','Keskuskatu 45','Helsinki','21240','Finland'),(91,'Wolski','Zbyszek','ul Filtrowa 68','Walla','01-012','Poland');
/*!40000 ALTER TABLE `customers` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `employees`
--

DROP TABLE IF EXISTS `employees`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `employees` (
  `EmployeeID` int NOT NULL AUTO_INCREMENT,
  `LastName` varchar(40) DEFAULT NULL,
  `FirstName` varchar(20) DEFAULT NULL,
  `BirthDate` date DEFAULT NULL,
  `Photo` varchar(100) DEFAULT NULL,
  `Notes` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`EmployeeID`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `employees`
--

LOCK TABLES `employees` WRITE;
/*!40000 ALTER TABLE `employees` DISABLE KEYS */;
INSERT INTO `employees` VALUES (1,'Davolio','Nancy','1968-12-08','EmpID1.pic','Education includes a BA in psychology from Colorado State University. She also completed (The Art of the Cold Call). Nancy is a member of Toastmasters International.'),(2,'Fuller','Andrew','1952-02-19','EmpID2.pic','Andrew received his BTS commercial and a Ph.D. in international marketing from the University of Dallas. He is fluent in French and Italian and reads German. He joined the company as a sales representative, was promoted to sales manager and was then named vice president of sales. Andrew is a member of the Sales Management Roundtable, the Seattle Chamber of Commerce, and the Pacific Rim Importers Association.'),(3,'Leverling','Janet','1963-08-30','EmpID3.pic','Janet has a BS degree in chemistry from Boston College). She has also completed a certificate program in food retailing management. Janet was hired as a sales associate and was promoted to sales representative.'),(4,'Peacock','Margaret','1958-09-19','EmpID4.pic','Margaret holds a BA in English literature from Concordia College and an MA from the American Institute of Culinary Arts. She was temporarily assigned to the London office before returning to her permanent post in Seattle.'),(5,'Buchanan','Steven','1955-03-04','EmpID5.pic','Steven Buchanan graduated from St. Andrews University, Scotland, with a BSC degree. Upon joining the company as a sales representative, he spent 6 months in an orientation program at the Seattle office and then returned to his permanent post in London, where he was promoted to sales manager. Mr. Buchanan has completed the courses Successful Telemarketing and International Sales Management. He is fluent in French.'),(6,'Suyama','Michael','1963-07-02','EmpID6.pic','Michael is a graduate of Sussex University (MA, economics) and the University of California at Los Angeles (MBA, marketing). He has also taken the courses Multi-Cultural Selling and Time Management for the Sales Professional. He is fluent in Japanese and can read and write French, Portuguese, and Spanish.'),(7,'King','Robert','1960-05-29','EmpID7.pic','Robert King served in the Peace Corps and traveled extensively before completing his degree in English at the University of Michigan and then joining the company. After completing a course entitled Selling in Europe, he was transferred to the London office.'),(8,'Callahan','Laura','1958-01-09','EmpID8.pic','Laura received a BA in psychology from the University of Washington. She has also completed a course in business French. She reads and writes French.'),(9,'Dodsworth','Anne','1969-07-02','EmpID9.pic','Anne has a BA degree in English from St. Lawrence College. She is fluent in French and German.'),(10,'West','Adam','1928-09-19','EmpID10.pic','An old chum.');
/*!40000 ALTER TABLE `employees` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `orders`
--

DROP TABLE IF EXISTS `orders`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `orders` (
  `OrderID` int NOT NULL,
  `CustomerID` int DEFAULT NULL,
  `EmployeeID` int DEFAULT NULL,
  `OrderDate` date DEFAULT NULL,
  `ShipperID` int DEFAULT NULL,
  PRIMARY KEY (`OrderID`),
  KEY `CustomerID` (`CustomerID`),
  KEY `EmployeeID` (`EmployeeID`),
  KEY `ShipperID` (`ShipperID`),
  CONSTRAINT `orders_ibfk_1` FOREIGN KEY (`CustomerID`) REFERENCES `customers` (`CustomerID`),
  CONSTRAINT `orders_ibfk_2` FOREIGN KEY (`EmployeeID`) REFERENCES `employees` (`EmployeeID`),
  CONSTRAINT `orders_ibfk_3` FOREIGN KEY (`ShipperID`) REFERENCES `shippers` (`ShipperID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `orders`
--

LOCK TABLES `orders` WRITE;
/*!40000 ALTER TABLE `orders` DISABLE KEYS */;
INSERT INTO `orders` VALUES (10248,90,5,'1996-07-04',3),(10249,81,6,'1996-07-05',1),(10250,34,4,'1996-07-08',2),(10251,84,3,'1996-07-08',1),(10252,76,4,'1996-07-09',2),(10253,34,3,'1996-07-10',2),(10254,14,5,'1996-07-11',2),(10255,68,9,'1996-07-12',3),(10256,88,3,'1996-07-15',2),(10257,35,4,'1996-07-16',3),(10258,20,1,'1996-07-17',1),(10259,13,4,'1996-07-18',3),(10260,55,4,'1996-07-19',1),(10261,61,4,'1996-07-19',2),(10262,65,8,'1996-07-22',3),(10263,20,9,'1996-07-23',3),(10264,24,6,'1996-07-24',3),(10265,7,2,'1996-07-25',1),(10266,87,3,'1996-07-26',3),(10267,25,4,'1996-07-29',1),(10268,33,8,'1996-07-30',3),(10269,89,5,'1996-07-31',1),(10270,87,1,'1996-08-01',1),(10271,75,6,'1996-08-01',2),(10272,65,6,'1996-08-02',2),(10273,63,3,'1996-08-05',3),(10274,85,6,'1996-08-06',1),(10275,49,1,'1996-08-07',1),(10276,80,8,'1996-08-08',3),(10277,52,2,'1996-08-09',3),(10278,5,8,'1996-08-12',2),(10279,44,8,'1996-08-13',2),(10280,5,2,'1996-08-14',1),(10281,69,4,'1996-08-14',1),(10282,69,4,'1996-08-15',1),(10283,46,3,'1996-08-16',3),(10284,44,4,'1996-08-19',1),(10285,63,1,'1996-08-20',2),(10286,63,8,'1996-08-21',3),(10287,67,8,'1996-08-22',3),(10288,66,4,'1996-08-23',1),(10289,11,7,'1996-08-26',3),(10290,15,8,'1996-08-27',1),(10291,61,6,'1996-08-27',2),(10292,81,1,'1996-08-28',2),(10293,80,1,'1996-08-29',3),(10294,65,4,'1996-08-30',2),(10295,85,2,'1996-09-02',2),(10296,46,6,'1996-09-03',1),(10297,7,5,'1996-09-04',2),(10298,37,6,'1996-09-05',2),(10299,67,4,'1996-09-06',2),(10300,49,2,'1996-09-09',2),(10301,86,8,'1996-09-09',2),(10302,76,4,'1996-09-10',2),(10303,30,7,'1996-09-11',2),(10304,80,1,'1996-09-12',2),(10305,55,8,'1996-09-13',3),(10306,69,1,'1996-09-16',3),(10307,48,2,'1996-09-17',2),(10308,2,7,'1996-09-18',3),(10309,37,3,'1996-09-19',1),(10310,77,8,'1996-09-20',2),(10311,18,1,'1996-09-20',3),(10312,86,2,'1996-09-23',2),(10313,63,2,'1996-09-24',2),(10314,65,1,'1996-09-25',2),(10315,38,4,'1996-09-26',2),(10316,65,1,'1996-09-27',3),(10317,48,6,'1996-09-30',1),(10318,38,8,'1996-10-01',2),(10319,80,7,'1996-10-02',3),(10320,87,5,'1996-10-03',3),(10321,38,3,'1996-10-03',2),(10322,58,7,'1996-10-04',3),(10323,39,4,'1996-10-07',1),(10324,71,9,'1996-10-08',1),(10325,39,1,'1996-10-09',3),(10326,8,4,'1996-10-10',2),(10327,24,2,'1996-10-11',1),(10328,28,4,'1996-10-14',3),(10329,75,4,'1996-10-15',2),(10330,46,3,'1996-10-16',1),(10331,9,9,'1996-10-16',1),(10332,51,3,'1996-10-17',2),(10333,87,5,'1996-10-18',3),(10334,84,8,'1996-10-21',2),(10335,37,7,'1996-10-22',2),(10336,60,7,'1996-10-23',2),(10337,25,4,'1996-10-24',3),(10338,55,4,'1996-10-25',3),(10339,51,2,'1996-10-28',2),(10340,9,1,'1996-10-29',3),(10341,73,7,'1996-10-29',3),(10342,25,4,'1996-10-30',2),(10343,44,4,'1996-10-31',1),(10344,89,4,'1996-11-01',2),(10345,63,2,'1996-11-04',2),(10346,65,3,'1996-11-05',3),(10347,21,4,'1996-11-06',3),(10348,86,4,'1996-11-07',2),(10349,75,7,'1996-11-08',1),(10350,41,6,'1996-11-11',2),(10351,20,1,'1996-11-11',1),(10352,28,3,'1996-11-12',3),(10353,59,7,'1996-11-13',3),(10354,58,8,'1996-11-14',3),(10355,4,6,'1996-11-15',1),(10356,86,6,'1996-11-18',2),(10357,46,1,'1996-11-19',3),(10358,41,5,'1996-11-20',1),(10359,72,5,'1996-11-21',3),(10360,7,4,'1996-11-22',3),(10361,63,1,'1996-11-22',2),(10362,9,3,'1996-11-25',1),(10363,17,4,'1996-11-26',3),(10364,19,1,'1996-11-26',1),(10365,3,3,'1996-11-27',2),(10366,29,8,'1996-11-28',2),(10367,83,7,'1996-11-28',3),(10368,20,2,'1996-11-29',2),(10369,75,8,'1996-12-02',2),(10370,14,6,'1996-12-03',2),(10371,41,1,'1996-12-03',1),(10372,62,5,'1996-12-04',2),(10373,37,4,'1996-12-05',3),(10374,91,1,'1996-12-05',3),(10375,36,3,'1996-12-06',2),(10376,51,1,'1996-12-09',2),(10377,72,1,'1996-12-09',3),(10378,24,5,'1996-12-10',3),(10379,61,2,'1996-12-11',1),(10380,37,8,'1996-12-12',3),(10381,46,3,'1996-12-12',3),(10382,20,4,'1996-12-13',1),(10383,4,8,'1996-12-16',3),(10384,5,3,'1996-12-16',3),(10385,75,1,'1996-12-17',2),(10386,21,9,'1996-12-18',3),(10387,70,1,'1996-12-18',2),(10388,72,2,'1996-12-19',1),(10389,10,4,'1996-12-20',2),(10390,20,6,'1996-12-23',1),(10391,17,3,'1996-12-23',3),(10392,59,2,'1996-12-24',3),(10393,71,1,'1996-12-25',3),(10394,36,1,'1996-12-25',3),(10395,35,6,'1996-12-26',1),(10396,25,1,'1996-12-27',3),(10397,60,5,'1996-12-27',1),(10398,71,2,'1996-12-30',3),(10399,83,8,'1996-12-31',3),(10400,19,1,'1997-01-01',3);
/*!40000 ALTER TABLE `orders` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ordersdetails`
--

DROP TABLE IF EXISTS `ordersdetails`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ordersdetails` (
  `OrderDetailID` int NOT NULL AUTO_INCREMENT,
  `OrderID` int DEFAULT NULL,
  `ProductID` int DEFAULT NULL,
  `Quantity` smallint DEFAULT NULL,
  PRIMARY KEY (`OrderDetailID`),
  KEY `OrderID` (`OrderID`),
  KEY `ProductID` (`ProductID`),
  CONSTRAINT `ordersdetails_ibfk_1` FOREIGN KEY (`OrderID`) REFERENCES `orders` (`OrderID`),
  CONSTRAINT `ordersdetails_ibfk_2` FOREIGN KEY (`ProductID`) REFERENCES `products` (`ProductID`)
) ENGINE=InnoDB AUTO_INCREMENT=200 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ordersdetails`
--

LOCK TABLES `ordersdetails` WRITE;
/*!40000 ALTER TABLE `ordersdetails` DISABLE KEYS */;
INSERT INTO `ordersdetails` VALUES (1,10248,11,12),(2,10248,42,10),(3,10248,72,5),(4,10249,14,9),(5,10249,51,40),(6,10250,41,10),(7,10250,51,35),(8,10250,65,15),(9,10251,22,6),(10,10251,57,15),(11,10252,20,40),(12,10252,33,25),(13,10252,60,40),(14,10253,31,20),(15,10253,39,42),(16,10253,49,40),(17,10254,24,15),(18,10254,55,21),(19,10254,74,21),(20,10255,2,20),(21,10255,16,35),(22,10255,36,25),(23,10255,59,30),(24,10256,53,15),(25,10256,77,12),(26,10257,27,25),(27,10257,39,6),(28,10257,77,15),(29,10258,2,50),(30,10258,5,65),(31,10258,32,6),(32,10259,21,10),(33,10259,37,1),(34,10260,41,16),(35,10260,57,50),(36,10260,62,15),(37,10260,70,21),(38,10261,21,20),(39,10261,35,20),(40,10262,5,12),(41,10262,7,15),(42,10262,56,2),(43,10263,16,60),(44,10263,24,28),(45,10263,30,60),(46,10263,74,36),(47,10264,2,35),(48,10264,41,25),(49,10265,17,30),(50,10265,70,20),(51,10266,12,12),(52,10267,40,50),(53,10267,59,70),(54,10267,76,15),(55,10268,29,10),(56,10268,72,4),(57,10269,33,60),(58,10269,72,20),(59,10270,36,30),(60,10270,43,25),(61,10271,33,24),(62,10272,20,6),(63,10272,31,40),(64,10272,72,24),(65,10273,10,24),(66,10273,31,15),(67,10273,33,20),(68,10273,40,60),(69,10273,76,33),(70,10274,71,20),(71,10274,72,7),(72,10275,24,12),(73,10275,59,6),(74,10276,10,15),(75,10276,13,10),(76,10277,28,20),(77,10277,62,12),(78,10278,44,16),(79,10278,59,15),(80,10278,63,8),(81,10278,73,25),(82,10279,17,15),(83,10280,24,12),(84,10280,55,20),(85,10280,75,30),(86,10281,19,1),(87,10281,24,6),(88,10281,35,4),(89,10282,30,6),(90,10282,57,2),(91,10283,15,20),(92,10283,19,18),(93,10283,60,35),(94,10283,72,3),(95,10284,27,15),(96,10284,44,21),(97,10284,60,20),(98,10284,67,5),(99,10285,1,45),(100,10285,40,40),(101,10285,53,36),(102,10286,35,100),(103,10286,62,40),(104,10287,16,40),(105,10287,34,20),(106,10287,46,15),(107,10288,54,10),(108,10288,68,3),(109,10289,3,30),(110,10289,64,9),(111,10290,5,20),(112,10290,29,15),(113,10290,49,15),(114,10290,77,10),(115,10291,13,20),(116,10291,44,24),(117,10291,51,2),(118,10292,20,20),(119,10293,18,12),(120,10293,24,10),(121,10293,63,5),(122,10293,75,6),(123,10294,1,18),(124,10294,17,15),(125,10294,43,15),(126,10294,60,21),(127,10294,75,6),(128,10295,56,4),(129,10296,11,12),(130,10296,16,30),(131,10296,69,15),(132,10297,39,60),(133,10297,72,20),(134,10298,2,40),(135,10298,36,40),(136,10298,59,30),(137,10298,62,15),(138,10299,19,15),(139,10299,70,20),(140,10300,66,30),(141,10300,68,20),(142,10301,40,10),(143,10301,56,20),(144,10302,17,40),(145,10302,28,28),(146,10302,43,12),(147,10303,40,40),(148,10303,65,30),(149,10303,68,15),(150,10304,49,30),(151,10304,59,10),(152,10304,71,2),(153,10305,18,25),(154,10305,29,25),(155,10305,39,30),(156,10306,30,10),(157,10306,53,10),(158,10306,54,5),(159,10307,62,10),(160,10307,68,3),(161,10308,69,1),(162,10308,70,5),(163,10309,4,20),(164,10309,6,30),(165,10309,42,2),(166,10309,43,20),(167,10309,71,3),(168,10310,16,10),(169,10310,62,5),(170,10311,42,6),(171,10311,69,7),(172,10312,28,4),(173,10312,43,24),(174,10312,53,20),(175,10312,75,10),(176,10313,36,12),(177,10314,32,40),(178,10314,58,30),(179,10314,62,25),(180,10315,34,14),(181,10315,70,30),(182,10316,41,10),(183,10316,62,70),(184,10317,1,20),(185,10318,41,20),(186,10318,76,6),(187,10319,17,8),(188,10319,28,14),(189,10319,76,30),(190,10320,71,30),(191,10321,35,10),(192,10322,52,20),(193,10323,15,5),(194,10323,25,4),(195,10323,39,4),(196,10324,16,21),(197,10324,35,70),(198,10324,46,30),(199,10324,59,40);
/*!40000 ALTER TABLE `ordersdetails` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `products`
--

DROP TABLE IF EXISTS `products`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `products` (
  `ProductID` int NOT NULL AUTO_INCREMENT,
  `ProductName` varchar(80) DEFAULT NULL,
  `SuppliersID` int DEFAULT NULL,
  `CategoryID` int DEFAULT NULL,
  `Unit` varchar(50) DEFAULT NULL,
  `Price` float DEFAULT NULL,
  PRIMARY KEY (`ProductID`),
  KEY `CategoryID` (`CategoryID`),
  KEY `SuppliersID` (`SuppliersID`),
  CONSTRAINT `products_ibfk_1` FOREIGN KEY (`CategoryID`) REFERENCES `categories` (`CategoryID`),
  CONSTRAINT `products_ibfk_2` FOREIGN KEY (`SuppliersID`) REFERENCES `suppliers` (`SupplierID`)
) ENGINE=InnoDB AUTO_INCREMENT=78 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `products`
--

LOCK TABLES `products` WRITE;
/*!40000 ALTER TABLE `products` DISABLE KEYS */;
INSERT INTO `products` VALUES (1,'Chais',1,1,'10 boxes x 20 bags',18),(2,'Chang',1,1,'24 - 12 oz bottles',19),(3,'Aniseed Syrup',1,2,'12 - 550 ml bottles',10),(4,'Chef Antons Cajun Seasoning',2,2,'48 - 6 oz jars',22),(5,'Chef Antons Gumbo Mix',2,2,'36 boxes',21.35),(6,'Grandmas Boysenberry Spread',3,2,'12 - 8 oz jars',25),(7,'Uncle Bobs Organic Dried Pears',3,7,'12 - 1 lb pkgs.',30),(8,'Northwoods Cranberry Sauce',3,2,'12 - 12 oz jars',40),(9,'Mishi Kobe Niku',4,6,'18 - 500 g pkgs.',97),(10,'Ikura',4,8,'12 - 200 ml jars',31),(11,'Queso Cabrales',5,4,'1 kg pkg.',21),(12,'Queso Manchego La Pastora',5,4,'10 - 500 g pkgs.',38),(13,'Konbu',6,8,'2 kg box',6),(14,'Tofu',6,7,'40 - 100 g pkgs.',23.25),(15,'Genen Shouyu',6,2,'24 - 250 ml bottles',15.5),(16,'Pavlova',7,3,'32 - 500 g boxes',17.45),(17,'Alice Mutton',7,6,'20 - 1 kg tins',39),(18,'Carnarvon Tigers',7,8,'16 kg pkg.',62.5),(19,'Teatime Chocolate Biscuits',8,3,'10 boxes x 12 pieces',9.2),(20,'Sir Rodneys Marmalade',8,3,'30 gift boxes',81),(21,'Sir Rodneys Scones',8,3,'24 pkgs. x 4 pieces',10),(22,'Gustafs Knäckebröd',9,5,'24 - 500 g pkgs.',21),(23,'Tunnbröd',9,5,'12 - 250 g pkgs.',9),(24,'Guaraná Fantástica',10,1,'12 - 355 ml cans',4.5),(25,'NuNuCa Nuß-Nougat-Creme',11,3,'20 - 450 g glasses',14),(26,'Gumbär Gummibärchen',11,3,'100 - 250 g bags',31.23),(27,'Schoggi Schokolade',11,3,'100 - 100 g pieces',43.9),(28,'Rössle Sauerkraut',12,7,'25 - 825 g cans',45.6),(29,'Thüringer Rostbratwurst',12,6,'50 bags x 30 sausgs.',123.79),(30,'Nord-Ost Matjeshering',13,8,'10 - 200 g glasses',25.89),(31,'Gorgonzola Telino',14,4,'12 - 100 g pkgs',12.5),(32,'Mascarpone Fabioli',14,4,'24 - 200 g pkgs.',32),(33,'Geitost',15,4,'500 g',2.5),(34,'Sasquatch Ale',16,1,'24 - 12 oz bottles',14),(35,'Steeleye Stout',16,1,'24 - 12 oz bottles',18),(36,'Inlagd Sill',17,8,'24 - 250 g jars',19),(37,'Gravad lax',17,8,'12 - 500 g pkgs.',26),(38,'Côte de Blaye',18,1,'12 - 75 cl bottles',263.5),(39,'Chartreuse verte',18,1,'750 cc per bottle',18),(40,'Boston Crab Meat',19,8,'24 - 4 oz tins',18.4),(41,'Jacks New England Clam Chowder',19,8,'12 - 12 oz cans',9.65),(42,'Singaporean Hokkien Fried Mee',20,5,'32 - 1 kg pkgs.',14),(43,'Ipoh Coffee',20,1,'16 - 500 g tins',46),(44,'Gula Malacca',20,2,'20 - 2 kg bags',19.45),(45,'Røgede sild',21,8,'1k pkg.',9.5),(46,'Spegesild',21,8,'4 - 450 g glasses',12),(47,'Zaanse koeken',22,3,'10 - 4 oz boxes',9.5),(48,'Chocolade',22,3,'10 pkgs.',12.75),(49,'Maxilaku',23,3,'24 - 50 g pkgs.',20),(50,'Valkoinen suklaa',23,3,'12 - 100 g bars',16.25),(51,'Manjimup Dried Apples',24,7,'50 - 300 g pkgs.',53),(52,'Filo Mix',24,5,'16 - 2 kg boxes',7),(53,'Perth Pasties',24,6,'48 pieces',32.8),(54,'Tourtière',25,6,'16 pies',7.45),(55,'Pâté chinois',25,6,'24 boxes x 2 pies',24),(56,'Gnocchi di nonna Alice',26,5,'24 - 250 g pkgs.',38),(57,'Ravioli Angelo',26,5,'24 - 250 g pkgs.',19.5),(58,'Escargots de Bourgogne',27,8,'24 pieces',13.25),(59,'Raclette Courdavault',28,4,'5 kg pkg.',55),(60,'Camembert Pierrot',28,4,'15 - 300 g rounds',34),(61,'Sirop dérable',29,2,'24 - 500 ml bottles',28.5),(62,'Tarte au sucre',29,3,'48 pies',49.3),(63,'Vegie-spread',7,2,'15 - 625 g jars',43.9),(64,'Wimmers gute Semmelknödel',12,5,'20 bags x 4 pieces',33.25),(65,'Louisiana Fiery Hot Pepper Sauce',2,2,'32 - 8 oz bottles',21.05),(66,'Louisiana Hot Spiced Okra',2,2,'24 - 8 oz jars',17),(67,'Laughing Lumberjack Lager',16,1,'24 - 12 oz bottles',14),(68,'Scottish Longbreads',8,3,'10 boxes x 8 pieces',12.5),(69,'Gudbrandsdalsost',15,4,'10 kg pkg.',36),(70,'Outback Lager',7,1,'24 - 355 ml bottles',15),(71,'Fløtemysost',15,4,'10 - 500 g pkgs.',21.5),(72,'Mozzarella di Giovanni',14,4,'24 - 200 g pkgs.',34.8),(73,'Röd Kaviar',17,8,'24 - 150 g jars',15),(74,'Longlife Tofu',4,7,'5 kg pkg.',10),(75,'Rhönbräu Klosterbier',12,1,'24 - 0.5 l bottles',7.75),(76,'Lakkalikööri',23,1,'500 ml',18),(77,'Original Frankfurter grüne Soße',12,2,'12 boxes',13);
/*!40000 ALTER TABLE `products` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `shippers`
--

DROP TABLE IF EXISTS `shippers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `shippers` (
  `ShipperID` int NOT NULL AUTO_INCREMENT,
  `ShipperName` varchar(30) DEFAULT NULL,
  `Phone` varchar(14) DEFAULT NULL,
  PRIMARY KEY (`ShipperID`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `shippers`
--

LOCK TABLES `shippers` WRITE;
/*!40000 ALTER TABLE `shippers` DISABLE KEYS */;
INSERT INTO `shippers` VALUES (1,'Speedy Express','(503) 555-9831'),(2,'United Package','(503) 555-3199'),(3,'Federal Shipping','(503) 555-9931');
/*!40000 ALTER TABLE `shippers` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `suppliers`
--

DROP TABLE IF EXISTS `suppliers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `suppliers` (
  `SupplierID` int NOT NULL AUTO_INCREMENT,
  `SuppliersName` varchar(80) DEFAULT NULL,
  `ContactName` varchar(30) DEFAULT NULL,
  `Address` varchar(80) DEFAULT NULL,
  `City` varchar(20) DEFAULT NULL,
  `PostalCode` varchar(10) DEFAULT NULL,
  `Country` varchar(20) DEFAULT NULL,
  `Phone` varchar(16) DEFAULT NULL,
  PRIMARY KEY (`SupplierID`)
) ENGINE=InnoDB AUTO_INCREMENT=30 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `suppliers`
--

LOCK TABLES `suppliers` WRITE;
/*!40000 ALTER TABLE `suppliers` DISABLE KEYS */;
INSERT INTO `suppliers` VALUES (1,'Exotic Liquid','Charlotte Cooper','49 Gilbert St.','Londona','EC1 4SD','UK','(171) 555-2222'),(2,'New Orleans Cajun Delights','Shelley Burke','P.O. Box 78934','New Orleans','70117','USA','(100) 555-4822'),(3,'Grandma Kellys Homestead','Regina Murphy','707 Oxford Rd.','Ann Arbor','48104','USA','(313) 555-5735'),(4,'Tokyo Traders','Yoshi Nagase','9-8 Sekimai Musashino-shi','Tokyo','100','Japan','(03) 3555-5011'),(5,'Cooperativa de Quesos Las Cabras','Antonio del Valle Saavedra','Calle del Rosal 4','Oviedo','33007','Spain','(98) 598 76 54'),(6,'Mayumis','Mayumi Ohno','92 Setsuko Chuo-ku','Osaka','545','Japan','(06) 431-7877'),(7,'Pavlova, Ltd.','Ian Devling','74 Rose St. Moonie Ponds','Melbourne','3058','Australia','(03) 444-2343'),(8,'Specialty Biscuits, Ltd.','Peter Wilson','29 Kings Way','Manchester','M14 GSD','UK','(161) 555-4448'),(9,'PB Knäckebröd AB','Lars Peterson','Kaloadagatan 13','Göteborg','S-345 67','Sweden','031-987 65 43'),(10,'Refrescos Americanas LTDA','Carlos Diaz','Av. das Americanas 12.890','São Paulo','5442','Brazil','(11) 555 4640'),(11,'Heli Süßwaren GmbH & Co. KG','Petra Winkler','Tiergartenstraße 5','Berlin','10785','Germany','(010) 9984510'),(12,'Plutzer Lebensmittelgroßmärkte AG','Martin Bein','Bogenallee 51','Frankfurt','60439','Germany','(069) 992755'),(13,'Nord-Ost-Fisch Handelsgesellschaft mbH','Sven Petersen','Frahmredder 112a','Cuxhaven','27478','Germany','(04721) 8713'),(14,'Formaggi Fortini s.r.l.','Elio Rossi','Viale Dante, 75','Ravenna','48100','Italy','(0544) 60323'),(15,'Norske Meierier','Beate Vileid','Hatlevegen 5','Sandvika','1320','Norway','(0)2-953010'),(16,'Bigfoot Breweries','Cheryl Saylor','3400 - 8th Avenue Suite 210','Bend','97101','USA','(503) 555-9931'),(17,'Svensk Sjöföda AB','Michael Björn','Brovallavägen 231','Stockholm','S-123 45','Sweden','08-123 45 67'),(18,'Aux joyeux ecclésiastiques','Guylène Nodier','203, Rue des Francs-Bourgeois','Paris','75004','France','(1) 03.83.00.68'),(19,'New England Seafood Cannery','Robb Merchant','Order Processing Dept. 2100 Paul Revere Blvd.','Boston','02134','USA','(617) 555-3267'),(20,'Leka Trading','Chandra Leka','471 Serangoon Loop, Suite #402','Singapore','0512','Singapore','555-8787'),(21,'Lyngbysild','Niels Petersen','Lyngbysild Fiskebakken 10','Lyngby','2800','Denmark','43844108'),(22,'Zaanse Snoepfabriek','Dirk Luchte','Verkoop Rijnweg 22','Zaandam','9999 ZZ','Netherlands','(12345) 1212'),(23,'Karkki Oy','Anne Heikkonen','Valtakatu 12','Lappeenranta','53120','Finland','(953) 10956'),(24,'Gday, Mate','Wendy Mackenzie','170 Prince Edward Parade Hunters Hill','Sydney','2042','Australia','(02) 555-5914'),(25,'Ma Maison','Jean-Guy Lauzon','2960 Rue St. Laurent','Montréal','H1J 1C3','Canada','(514) 555-9022'),(26,'Pasta Buttini s.r.l.','Giovanni Giudici','Via dei Gelsomini, 153','Salerno','84100','Italy','(089) 6547665'),(27,'Escargots Nouveaux','Marie Delamare','22, rue H. Voiron','Montceau','71300','France','85.57.00.07'),(28,'Gai pâturage','Eliane Noz','Bat. B 3, rue des Alpes','Annecy','74000','France','38.76.98.06'),(29,'Forêts dérables','Chantal Goulet','148 rue Chasseur','Ste-Hyacinthe','J2S 7S8','Canada','(514) 555-2955');
/*!40000 ALTER TABLE `suppliers` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2023-10-01 17:24:59
