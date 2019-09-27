.PHONY: clean

gjk_wrapper.so: gjk_wrapper.o gjk.o
	gcc -shared -o $@ $^

gjk_wrapper.o: gjk_wrapper.c gjk.h
	gcc -o $@ -c -Wall -fpic $<

gjk.o: gjk.c gjk.h
	gcc -o $@ -c -Wall -fpic $<

clean:
	-@rm *.so
	-@rm *.o
