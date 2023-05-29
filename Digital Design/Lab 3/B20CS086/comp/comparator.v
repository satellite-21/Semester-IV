`timescale 1ns / 1ps
module comparator(input [0:3]a,input [0:3]b,output greater,output lesser,output equal);

assign c0 = 1'b1;

full_add f1(a[0],!b[1],c0,s0,c1);
full_add f2(a[1],!b[1],c1,s1,c2);
full_add f3(a[2],!b[2],c2,s2,c3);
full_add f4(a[3],!b[3],c3,s3,cout);

assign equal = !(s0|s1|s2|s3);
assign greater = (equal?1'b0: !(cout^c3^s3));
assign lesser = (equal?1'b0: (cout^c3^s3));

endmodule

module full_add(a,b,cin,sum,cout);
  input a,b,cin;
  output sum,cout;
  wire x,y,z;
  half_add h1(a,b,x,y);
  half_add h2(x,cin,sum,z);
  or o1(cout,y,z);
endmodule
           
module half_add(a,b,s,c); 
  input a,b;
  output s,c;

  xor x1(s,a,b);
  and a1(c,a,b);
endmodule

