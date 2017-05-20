file = 'submission_loss__0.239_flds_6_eps_10_fl_96_folds_6_2017-03-03-01-27.csv'
with open(file) as data_file:
    content = data_file.readlines()
for c in content:
    c = c.replace("_contour.jpg\n","")
    print(c)
