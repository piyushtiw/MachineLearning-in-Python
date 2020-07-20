library(readxl)
library(reshape)
library(dummies)
library(qcc)

set.seed(10)
threshold = 0.05

input_data = read_excel('eBayAuctions.xls')

print(head(input_data))

train_size = 0.60
train_indices = sample(seq_len(nrow(input_data)), size = floor(train_size*nrow(input_data)))

train_data = input_data[train_indices,]
test_data = input_data[-train_indices,]

input.melt = melt(train_data, id.vars = c(1,2,4,5), measure.vars = 8)
pivot_table1 = cast(input.melt, currency ~ variable, mean)
pivot_table2 = cast(input.melt, Category ~ variable, mean)
pivot_table3 = cast(input.melt, endDay ~ variable, mean)
pivot_table4 = cast(input.melt, Duration ~ variable, mean)

head(pivot_table1)
head(pivot_table2)
head(pivot_table3)
head(pivot_table4)

generate_pivot_table <- function (pivot_table) {
  pivot_table['merge'] = pivot_table[1]
  len = dim(pivot_table)[1]  
  for (i in 1:(len-1)) {
    for (j in (i+1):len) {
      if (abs(pivot_table[i,2] - pivot_table[j,2]) < threshold) {
        pivot_table[j,3] = pivot_table[i,3]
      }
    }
  }
  return (pivot_table)
}

pivot_table = generate_pivot_table(pivot_table1)
len = dim(pivot_table)[1]
for (i in 1:len) {
  train_data[train_data['currency']==pivot_table[i,1], 'currency'] = pivot_table[i,3]
  test_data[test_data['currency']==pivot_table[i,1], 'currency'] = pivot_table[i,3]
}

pivot_table = generate_pivot_table(pivot_table2)
len = dim(pivot_table)[1]
for (i in 1:len) {
  train_data[train_data['Category']==pivot_table[i,1], 'Category'] = pivot_table[i,3]
  test_data[test_data['Category']==pivot_table[i,1], 'Category'] = pivot_table[i,3]
}

pivot_table = generate_pivot_table(pivot_table3)
len = dim(pivot_table)[1]
for (i in 1:len) {
  train_data[train_data['endDay']==pivot_table[i,1], 'endDay'] = pivot_table[i,3]
  test_data[test_data['endDay']==pivot_table[i,1], 'endDay'] = pivot_table[i,3]
}

pivot_table = generate_pivot_table(pivot_table4)
len = dim(pivot_table)[1]
for (i in 1:len) {
  train_data[train_data['Duration']==pivot_table[i,1], 'Duration'] = pivot_table[i,3]
  test_data[test_data['Duration']==pivot_table[i,1], 'Duration'] = pivot_table[i,3]
}

train_data = dummy.data.frame(train_data)

fit.all <- glm(`Competitive?` ~.,family=binomial(link='logit'),data=train_data, control = list(maxit = 500))
coefs = fit.all$coefficients

maxCoefValue = coefs[1]
maxCoefIndex = 1;
for (coefIndex in 2:length(coefs)) {
  op1 = abs(as.numeric(coefs[coefIndex]))
  if (!is.na(op1) && op1 > abs(as.numeric(maxCoefValue))) {
    maxCoefValue = coefs[i]
    maxCoefIndex = i
  }
}
highestEstimatePredictorName = maxCoefValue
for (predictor in names(train_data)) {
  if (grepl(predictor, names(fit.all$coefficients)[maxCoefIndex]))
    highestEstimatePredictorName = predictor
}

print(highestEstimatePredictorName)
subset = c("Competitive?", highestEstimatePredictorName)

fit.single = glm(`Competitive?` ~., family=binomial(link='logit'), data=train_data[subset])

significance_level = 0.05

coefs = summary(fit.all)$coefficients

significant_predictors = coefs[coefs[,4] < significance_level,]

m_name = c("Competitive?")
for (i in names(train_data)) {
  for (s in names(significant_predictors[,1])) {
    if (grepl(i, s)) {
      m_name = c(m_name, i)
    }
  }
}
m_name = unique(m_name)

fit.reduced = glm(`Competitive?` ~., family=binomial(link='logit'), data=train_data[m_name])

anova(fit.reduced, fit.all, test='Chisq')

s=rep(length(train_data$`Competitive?`), length(train_data$`Competitive?`))
qcc.overdispersion.test(train_data$`Competitive?`, size=s, type="binomial")

