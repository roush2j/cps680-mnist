
mnist.jar: bin
	jar cfe mnist.jar edu.cmich.cps680fall2016.mnist.Main src data -C bin ./

bin: src/edu/cmich/cps680fall2016/mnist/*.java
	mkdir -p bin
	touch bin
	javac -d bin src/edu/cmich/cps680fall2016/mnist/*.java
