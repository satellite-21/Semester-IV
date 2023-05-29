`timescale 1ns / 1ps
module bcd_to_sevenseg(bcd,display);

    input[0:2]bcd;
    output[0:6]display;
    
    reg [0:6]display;
    
        always@(bcd)
            case(bcd)
                
                0 : display = 7'b1111110;
                1 : display = 7'b0110000;
                2 : display = 7'b1101101;
                3 : display = 7'b1111001;
                4 : display = 7'b0110011;
                5 : display = 7'b1011011;
                6 : display = 7'b1011111;
                7 : display = 7'b1110000;
                8 : display = 7'b1111111;
                9 : display = 7'b1111011;
                
            default : display = 7'b0000000;
            endcase
endmodule

module half_adder(a,b,s,c);
input a,b;
output s,c;

assign s = a^b;
assign c = a&b;

endmodule


module full_adder(A,B,C,s_ans,c_ans);
input A,B,C;
output s_ans,c_ans;

half_adder H1(B,C,s1,c1);
half_adder H2(A,s1,s2,c2);

assign s_ans = s2;
assign c_ans = c1|c2;

endmodule

module multiplier(a, b, ans,d1,d2);

input   [2:0] a, b;
output [5:0] ans;
output [6:0] d1,d2;

wire [7:0] c,c1;
wire [3:0] temp;
assign c[0] = 1'b0;

assign ans[0] = a[0]&b[0];

full_adder f1(a[1]&b[0],a[0]&b[1],c[0],ans[1],c[1]);

full_adder f2(a[2]&b[0],a[1]&b[1],c[1],c1[0],c1[1]);
full_adder f3(c1[0],a[0]&b[2],c[0],ans[2],temp[1]);

half_adder h1(c1[1],temp[1],c[3],c1[2]);
full_adder f4(a[1]&b[2],a[2]&b[1],c[3],ans[3],temp[2]);

half_adder h2(c1[2],temp[2],c[4],c1[3]);
full_adder f5(a[2]&b[2],c[0],c[4],ans[4],temp[3]);

assign ans[5] = temp[3]|c1[3];

bcd_to_sevenseg bcd71(ans[5:3],d1);
bcd_to_sevenseg bcd72(ans[2:0],d2);

endmodule
