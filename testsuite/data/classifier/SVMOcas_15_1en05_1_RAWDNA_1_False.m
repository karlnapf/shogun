classifier_name = 'SVMOcas';
classifier_C = 1.5;
classifier_epsilon = 1e-05;
init_random = 42;
classifier_data_train = {'GATCCGTCGGCTTGTTGTGGGTATTATACGATTGACGTGCGGTATGTCGTATGACGAGCG', 'ATTTGATAACCTGCCTCCATGAAGGCTACTCGCCTCTAAAGCTCTATTTTTGAATGTCAA', 'GCACAGCTCCAACCGCGTTACAATACGGGACGACAACAGGTAAAGGATAGGTGTCTATGT', 'CGGGAAAATATCATGCGGATTGAGAACCTGGCCCTATGCGCTTCCGTAGGGAGTAATATC', 'ATCGCGACTCGCTATCAAAGCTCAGGAAAGGCTTGTAACAGTAACTCGAATTATGTCGCG', 'AACGAGTAGCTAGTTGCGTGATGGAGCGAGTATCTTGGTCCGTTTCTAGACCCGCCACAA', 'GGGCCTGCACCCGGGTTGTAAAGTATATCACCCCTGCGGCCAAGAGGAGCTCTGAGTGTT', 'AAGCTTCATCTCGAGCGAGTAAAGGGACGCTAGCCGGTAAGGCAATACACAGATAGTGGC', 'CGTGGGCAGTAGAGACTCGTTACATTGTGGATATTGACCCATGCCCGGTCTTATATCTTT', 'TCACAGCTGTATTCGTAGCACAAATGTCCGCAGAGGAATCATAGTAAACAGAGTTGCGGC', 'ATCACCTGTACCACTAGAAATTTCAGCTACACGCCCCTTTTCGCCTATCGGTGGAGAGGG'};
classifier_num_threads = 1;
classifier_label_type = 'twoclass';
classifier_data_test = {'AAGAAATGCCCTGAAAGTCAATCTGCAAGTGGATTCTGGGTGACTACCCGTAAATGCTTT', 'TCTCTGTGAGTTAAAGTAGTGGGGCGGTGTACAAGTACTAGGGACTGATAGATCCACTTC', 'CGTGCCGAGCTAGCGAGGGGGTCTCAACTGTGCACCAAGGTTGCGTACTCGGACCAATCG', 'CTCCTTATCCAAGCCGTAGATGTAACCATACTCTACAATAGCCTAGTGTACCGTCGAGCT', 'GCAAACTGGATGAGGACCATTGCCGGAGTCATACACTGATTTACTTAACTGCAGTCTTAG', 'CTGGTCTGAGTCTTCTAAATGACGATTGTCTGGATCAAGGGTAGATTGGCCTCCCTCAGT', 'AAACCGGGCTCTGAGCGAATATCGGGCAACGTTTTGCTGGACCTGCTACAGTCTCTTTAT', 'GCTCTAGGAAACCGCCGATTGGTAAATAGTTGATGAATGTTGGCCCTGTCGGGGAGACCG', 'TGGGGACCCAACTCCAGCAGAAAAACTGGCTCAGACGTCGTACCCCTGGCTTTAGACAGG', 'ACCCACTAACTGGCGAGTGTGATAGAACCCGAAGCGGTGCGCCTCTTTTCCGCGGGCACA', 'TCAAGGCACCGGCATCGACAGTCAACAATCCTATCTGTTCTCCCGAGGTTCCCAACCTGC', 'GACGCCTTGGGACACGATTGGTATATCTTACTCTGCCATCGCGTTCTTGTAATCTTATCG', 'GTACGGGACCTCATATGTCAGTGATTGTTGACACCACATTATGCAGTGAACCACTTCTCC', 'AGTGCGATGCGGAGAGTTGGATCCCAGAGCAGACGATAAGCACCTCTTCTCTAGCCGGAA', 'GGTTTGCATGACTGCCACCATCTTAGCTAGGCAGGCGCGATCCAAGCTACCAACCCGGAC', 'TGGCTTGTAATTACACTGCCACGTTATTTCCCAAACGCATTTTGTACAAAGTGATGCGCC', 'TCCTGAAAGCATGAGGACTCGCAACAATTTTATATTGTGCCGAGTATTTTGTTACGAGGA'};
classifier_labels = [1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1];
classifier_accuracy = 0.0001;
classifier_alphabet = 'RAWDNA';
classifier_order = 1;
classifier_feature_type = 'Byte';
classifier_bias_enabled = 'False';
classifier_type = 'linear';
classifier_classified = [0.0777466282, -0.175165905, -0.228155538, -0.240319843, -0.362544854, -0.215238922, -0.517978708, 0.0463987816, -0.112493855, -0.200725623, -0.334213032, 0.0666691137, -0.187680197, 0.0626811672, -0.266336685, -0.11726091, -0.0544071142];
classifier_feature_class = 'wd';
