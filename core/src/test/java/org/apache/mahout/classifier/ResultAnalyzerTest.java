/*
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

package org.apache.mahout.classifier;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public class ResultAnalyzerTest extends MahoutTestCase {
  
  @Test
  public void toyData() throws Exception {
    Map<Integer,String> labelMap = new HashMap<Integer,String>();
    labelMap.put(0, "label1");
    labelMap.put(1, "label2");
    ResultAnalyzer analyzer = new ResultAnalyzer(labelMap.values(), "DEFAULT");
    
    ClassifierResult classifierResult = new ClassifierResult("label1", 0.7);
    analyzer.addInstance("label1", classifierResult);
    classifierResult = new ClassifierResult("label1", 0.9);
    analyzer.addInstance("label1", classifierResult);
    classifierResult = new ClassifierResult("label1", 0.8);
    analyzer.addInstance("label2", classifierResult);
    classifierResult = new ClassifierResult("label2", 0.9);
    analyzer.addInstance("label2", classifierResult);
    
    ConfusionMatrix confusionMatrix = analyzer.getConfusionMatrix();
    assertEquals(confusionMatrix.getCount("label1", "label1"), 2);
    assertEquals(confusionMatrix.getCount("label1", "label1"), 2);
    assertEquals(confusionMatrix.getCount("label2", "label1"), 1);
    assertEquals(confusionMatrix.getCount("label1", "label2"), 0);
    assertEquals(confusionMatrix.getCount("label2", "label2"), 1);
    
    Pattern pattern = Pattern.compile(
        "Correctly Classified Instances[:\\t ]+3[\\t ]+75%",
        Pattern.CASE_INSENSITIVE);
    Matcher matcher = pattern.matcher(analyzer.toString());
    assertTrue(matcher.find());
  }
}
