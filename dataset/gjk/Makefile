OPTIM=-O3
.PHONY: clean

gjk_wrapper.so: gjk_wrapper.o gjk.o 
	gcc -shared -fpic -o $@ $^ $(OPTIM)

gjk_wrapper.o: gjk_wrapper.c gjk.h Makefile
	gcc -o $@ -c -Wall -fpic $< $(OPTIM)

gjk.o: gjk.c gjk.h Makefile
	gcc -o $@ -c -Wall -fpic $< $(OPTIM)

clean:
	-@rm *.so
	-@rm *.o
