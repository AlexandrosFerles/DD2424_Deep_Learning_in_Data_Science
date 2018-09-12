with open('arch_search_4_layers.txt','r') as arch_search:

	for line in arch_search:

		if '-' in line:
			continue
		else:
			sec = float(line.split(':')[1][1:].split('\n')[0])
			if sec >= 43.5:
				print (line)
