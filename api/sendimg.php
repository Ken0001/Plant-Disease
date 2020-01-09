<?php
    header("Content-type:text/html;charset=utf-8");

    $a = 5;
    $p = "/home/divc/Desktop/plant-project/api/predict.py";
    $data = "/home/divc/Desktop/plant-project/dataset/original/test/black_2.jpg";
    $model = "/home/divc/Desktop/plant-project/model/densenet.h5";
    $out = shell_exec("/usr/bin/python3 $p -d=$data -m=$model");
    var_dump($out);
   
?>