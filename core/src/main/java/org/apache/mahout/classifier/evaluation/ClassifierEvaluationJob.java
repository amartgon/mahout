/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.evaluation;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads the output of a classifier tester job and prints evaluation statistics.
 * Takes input in {@code SequenceFile<Text,VectorWritable>} format. Prints
 * summary statistics (number and percentage of correctly and incorrectly
 * classified instances) and the confusion matrix.
 */
public class ClassifierEvaluationJob extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(ClassifierEvaluationJob.class);
  
  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOption("labelIndex", "l", "The path to the location of the label index", true);
    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }
    
    // load the labels
    Map<Integer,String> labelMap = BayesUtils.readLabelIndex(getConf(), new Path(
        getOption("labelIndex")));
    
    // loop over the results and create the confusion matrix
    SequenceFileDirIterable<Text,VectorWritable> dirIterable = 
        new SequenceFileDirIterable<Text,VectorWritable>(getInputPath(),
                                                         PathType.LIST,
                                                         PathFilters.partFilter(),
                                                         getConf());
    ResultAnalyzer analyzer = new ResultAnalyzer(labelMap.values(), "DEFAULT");
    analyzeResults(labelMap, dirIterable, analyzer);
    log.info("Classification results:\n{}", analyzer);
    return 0;
  }
  
  private static void analyzeResults(Map<Integer,String> labelMap,
                                     SequenceFileDirIterable<Text,VectorWritable> dirIterable,
                                     ResultAnalyzer analyzer) {
    for (Pair<Text,VectorWritable> pair : dirIterable) {
      int bestIdx = Integer.MIN_VALUE;
      double bestScore = Long.MIN_VALUE;
      for (Vector.Element element : pair.getSecond().get()) {
        if (element.get() > bestScore) {
          bestScore = element.get();
          bestIdx = element.index();
        }
      }
      if (bestIdx != Integer.MIN_VALUE) {
        ClassifierResult classifierResult = new ClassifierResult(labelMap.get(bestIdx), bestScore);
        analyzer.addInstance(pair.getFirst().toString(), classifierResult);
      }
    }
  }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ClassifierEvaluationJob(), args);
  }
}
