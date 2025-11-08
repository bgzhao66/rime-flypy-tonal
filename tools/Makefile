export PYTHONHASHSEED=0

PRIMARY_NAME = flypy_tonal

.SILENT:
all:
	python3 ./convert_to_flypy_tonal.py --name $(PRIMARY_NAME) --chinese_code  --phrase --abbreviate \
				--extra_table --difference 	\
				pinyin_trad_ext1.dict.txt   \
				extra/essay.txt
#				extra/360wdict.txt    \
#				extra/12dicts.txt     \
#				extra/words.txt       \
#				extra/luna_classic.txt

	echo "All tables converted"

mapping:
	python3 ./mapping_bopomofo.py > mapping.txt

.PHONY: clean
clean:
	rm -f ../$(PRIMARY_NAME)*.dict.yaml
