.PHONY: cpp_code

cpp_code:
	make -C cpp

clean:
	make -C cpp clean
	rm -f *.so
	rm -f ctools*
