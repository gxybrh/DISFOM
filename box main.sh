#！/bin/bash
for i in {14..14} # 7..14
do
	python3 qua_box.py $((2 ** i))
done
# python3 qua_main.py 16384
# python3 qua_main.py 100
# python3 qua_main.py 316
# python3 qua_main.py 1000
# python3 qua_main.py 3162
# python3 qua_main.py 10000
# python3 qua_main.py 31623