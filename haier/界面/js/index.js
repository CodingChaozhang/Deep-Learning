option1 = {
        tooltip : {
            trigger: 'axis',
            axisPointer : {            // 坐标轴指示器，坐标轴触发有效
                type : 'shadow'        // 默认为直线，可选为：'line' | 'shadow'
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true,
            textStyle:{
                color:'#c1c1c1'
            }
        },
        xAxis:  {
            type: 'value',
            axisLabel: {  
                show: true,
                interval: 'auto',  
                formatter: '{value}%',
                textStyle:{
                    color:'#c1c1c1'
                }
              },
              axisLine: {
                  lineStyle: {
                      color: '#c1c1c1',//左边线的颜色
                  }
              },
        },
        yAxis: {
            type: 'category',
            axisLabel: { 
                textStyle:{
                    color:'#c1c1c1'
                }
              },
              axisLine: {
                  lineStyle: {
                      color: '#c1c1c1',//左边线的颜色
                  }
              },
            data: ['亮度指数','光滑指数','平整度','面积比']
        },
        series: [
            {
                name: '成功率',
                type: 'bar',
                stack: '总量',
                label: {
                    normal: {
                        show: true,
                        position: 'insideRight',
                        formatter: '{c}%',
                    }
                },
                itemStyle:{  
                    normal:{  
                        color:function(params) {
                            //自定义颜色
                            var colorList = [          
                                    '#d04f39ba', '#00FFFFba','#efb32dba', '#4ad01eba'
                                ];
                                return colorList[params.dataIndex]
                             }
                    }  
               },
                data: [91, 92, 93, 94]
            }
        ]
    };


option2 = {
        tooltip: {
            trigger: 'item',
            formatter: "{a} <br/>{b}: {c} ({d}%)"
        },
        // legend: {
        //     orient: 'vertical',
        //     x: 'left',
        //     data:['直接访问','邮件营销','联盟广告','视频广告','搜索引擎']
        // },
        series: [
            {
                name:'良品率',
                type:'pie',
                radius: ['50%', '70%'],
                avoidLabelOverlap: false,
                label: {
                    normal: {
                        show: false,
                        position: 'center'
                    },
                    emphasis: {
                        show: true,
                        textStyle: {
                            fontSize: '30',
                            fontWeight: 'bold'
                        }
                    }
                },
                labelLine: {
                    normal: {
                        show: false
                    }
                },
                data:[
                    {value:97, name:'成功率'},
                    {value:3, name:'失败率'}
                ],
                itemStyle: {
                        normal:{
                            color:function(params) {
                            //自定义颜色
                            var colorList = [          
                                    '#00FFFFba', '#d04f39ba'
                                ];
                                return colorList[params.dataIndex]
                             }
                        }
                }
            }
        ]
    };

    
    function randomData() {
        now = new Date();
        value = Math.round(Math.random()*40);
        while(value==0){
            value = Math.round(Math.random()*40);
        }
        data.name.push(now.getHours()+':'+now.getMinutes()+':'+now.getSeconds()+'.'+now.getMilliseconds());
        data.value.push(value);
    }

            
    var data ={name:[],value:[]};
    data.name = new Array(50);
    data.value = new Array(50);
    
    option3 = {
        // title: {
        //     text: '动态数据 + 时间坐标轴'
        // },
        tooltip: {
            trigger: 'axis',        
            formatter: function (params) {
                params = params[0];
                return params.value;
            },
            axisPointer: {
                animation: true
            }
        },
        xAxis: {
            data : data.name,
            axisLine: {
                lineStyle: {
                    color: '#c1c1c1',//左边线的颜色
                }
            },
            axisLabel: { 
                textStyle:{
                    color:'#c1c1c1'
                }
              },
            splitLine: {
                show: false
            }
        },
        
        yAxis: {
            type: 'value',
            axisLine: {
                lineStyle: {
                    color: '#c1c1c1',//左边线的颜色
                }
            },
            axisLabel: { 
                textStyle:{
                    color:'#c1c1c1'
                }
              },
            splitLine: {
                show: false
            },
            max:60
        },
        series: [{
            name: '数据',
            type: 'line',
            showSymbol: false,
            hoverAnimation: false,
            data: data.value,
            itemStyle:{  
                normal:{  
                    color:'#d04f39fc'
                }  
           },
           markLine:{
            itemStyle:{
                normal:{
                    lineStyle:{
                        type:'solid',
                        color:'#efb32dba'
                    },
                    label:{
                        show:false
                    }
                }
            },
            symbol:'none',
            large:true,
            data: [
                {
                    name: '水平线',
                    yAxis: 50,
                    type: 'average'
                },
            ]
        },
            
        }]
    };

    //chart1随机
    setInterval(function () {
        var lzw1=Math.round(Math.random()*10)+90;
        var lzw2=Math.round(Math.random()*10)+90;
        var lzw3=Math.round(Math.random()*10)+90;
        var lzw4=Math.round(Math.random()*10)+90;
        var data2=[lzw1, lzw2, lzw3, lzw4];
        myChart1.setOption({
            series: [{
                data: data2
            }]
        });
    }, 100);

    //chart2随机
    setInterval(function () {
        var lzw=Math.round(Math.random()*5)+95;
        var data2=[
            {value:lzw, name:'成功率'},
            {value:(100-lzw), name:'失败率'}
        ]
        myChart2.setOption({
            series: [{
                data: data2
            }]
        });
        $(".chart2-label").text(lzw+"%");
    }, 35000);


    //chart3随机
    setInterval(function () {
        data.name.shift();
        data.value.shift();
        randomData();

        myChart3.setOption({
        series: [{
            data: data.value
        }],
        xAxis: {
        data: data.name,
        }
        });
        console.log(data);
    }, 100);


    



var myChart1;
var myChart2;
var myChart3;
addChart1();
addChart2();
addChart3();


function addChart1(){ 
    myChart1 = echarts.init(document.getElementById('chart1'));
    myChart1.setOption(option1);
}

function addChart2(){
    myChart2 = echarts.init(document.getElementById('chart2'), null, {renderer: 'svg'});
    myChart2.setOption(option2);
}

function addChart3(){
    myChart3 = echarts.init(document.getElementById('chart3'));
    if (option3 && typeof option3 === "object") {
        myChart3.setOption(option3, true);
    }
}

var all = 134;
var good_count = 134;
var working_time = 12;

//总数与良品数
setInterval(function () {
    $("#all").text((++all)+"个");
    $("#good_count").text((++good_count)+"个");
}, 35000);

//运行时间
setInterval(function () {
    $("#working_time").text((++working_time)+"min");
}, 60000);
  
