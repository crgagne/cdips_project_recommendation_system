for brand in 'euro' 'zimmur' 'angel' 'tigrisso' 'pierre' 'tory' 'w' 'motoley' 'medo' 'rubin' 'jiuzi' 'kisscat' 'eminihouse' 'here' 'sc' 'sj' 'oece' 'ms' 'moco' 'masfer' 'lesele' 'ladyangel' 'kldm' 'jwm' 'jefen' 'baiyi'
do
	python ex3_batch.py $brand | awk -F\| 'NF==3' | sort -t\| -k1,1 -k3,3nr | python page_gen.py $brand > img/index_$brand.html
done
