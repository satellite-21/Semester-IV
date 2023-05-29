`timescale 1ns / 1ps

module bcdSubtractor(a,b,c,s);

input [3:0] a;
input [3:0] b;
output [3:0] c;
output s;
    wire [3:0] tenscomp, tsum, tsum2;
    wire ct;
    tensComp t(b, tenscomp);
    bcdAdder A(a, tenscomp, 1'b0, tsum, ct);
    assign s = !ct;
    tensComp t2(tsum, tsum2);
    assign c = s?tsum2:tsum;
endmodule

module tensComp(a,b);

input [0:3]a;
output [0:3]b;

assign b = 10 -a;

endmodule

module bcdAdder(input [3:0]a,input [3:0]b,input carryInput,output reg [3:0]sum,output reg carry);

    reg [4:0] sumTemp;
    
    always @(a,b,carryInput)
    begin
        sumTemp = a+b+carryInput; 
        if(sumTemp > 9)    
            begin
            sumTemp = sumTemp+6; 
            carry = 1;  
            sum = sumTemp[3:0];    
            end
        else    
            begin
            carry = 0;
            sum = sumTemp[3:0];
            end
    end

endmodule

