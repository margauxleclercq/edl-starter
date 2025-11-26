# Makefile pour Money
# Usage: make test

JAVAC = javac
JAVA = java
CP = .;../lib/junit-4.13.2.jar;../lib/hamcrest-core-1.3.jar

test:
	$(JAVAC) -cp $(CP) *.java
	$(JAVA) -cp $(CP) org.junit.runner.JUnitCore MoneyTest
	rm *.class

clean:
	rm -f *.class
