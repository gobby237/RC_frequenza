const int N=65;
//=====================================================
double GetCHIS(double fft=0, TGraphErrors* gp=0, int opt=1, int var=1){
  TF1* fun;
  if(var==1){
    if(opt==1){fun = new TF1("fun",Form("1/TMath::Sqrt(1+(x/%f)^2)",fft),1e-3,1e4);}
    if(opt==2){fun = new TF1("fun",Form("1/TMath::Sqrt(1+(%f/x)^2)",fft),1e-3,1e4);}
  }
  if(var==2){
    if(opt==1){fun = new TF1("fun",Form("(2./TMath::Pi())*TMath::ATan(x/%f)",fft),1e-3,1e4);}
    if(opt==2){fun = new TF1("fun",Form("(2./TMath::Pi())*TMath::ATan(%f/x)",fft),1e-3,1e4);}
  }
  double val=0;
  for(int i=0; i<N; i++){
    double erry=gp->GetErrorY(i);
    double xx=0;
    double yy=0;
    gp->GetPoint(i,xx,yy);
    double res=yy-fun->Eval(xx);
    if(erry>0){
      val+=pow(res/erry,2);
    }
  }
  return val;
}
//=====================================================
void RCF(int opt=1, int opt2=999, double Cmeas=9.36e-9, double V0=7.36){

  //double Cmeas=21e-9;//21 nF Agilent 10 00 nF
  //double Rmeas=3.846e3;//3.846 kOhm Agilent 6 kOhm

  //double Cmeas=99e-9;//Agilent
  double Rmeas=3.871e3;//Agilent

  double fl0=2.2;
  double fl1=6.4;
  double fitfl0=2.5;
  double fitfl1=6.7;
  double amin=0.3;
  double amax=0.9;

  char* lab="";
  if(opt==1)lab="PB";
  if(opt==2)lab="PA";

  char* infil="";

  if(opt==1)infil=Form("RCF_PB_id%02d.txt",opt2);
  if(opt==2)infil=Form("RCF_PA_id%02d.txt",opt2);
  
  if(opt2==999){
    if(opt==1)infil="RCF_PB.txt";
    if(opt==2)infil="RCF_PA.txt";
  }
  
  if (opt2==-13){
    if(opt==1)infil="RCF_PB_R22ohm_id13.txt";
    if(opt==2)infil="RCF_PA_R22ohm_id13.txt";
    Rmeas=22.0;
    fl0=20;
    fl1=200;
    fitfl0=70;
    fitfl1=90;
    amin=0;
    amax=1;
  }

  //cout<<infil<<endl;
  //return;
  
  double eCmeas=TMath::Sqrt(pow((0.58*1.9/100.)*Cmeas,2)+pow(0.58*2e-9,2));
  double eRmeas=TMath::Sqrt(pow((0.58*0.9/100.)*Rmeas,2)+pow(0.58*0.003e3,2));
  if (opt2==-13){
    Cmeas=94e-9;
    eRmeas=TMath::Sqrt(pow((0.58*0.9/100.)*Rmeas,2)+pow(0.58*0.3,2));
  }
  double ftexp=(1./(2*TMath::Pi()*Rmeas*Cmeas));
  double eftexp=ftexp*TMath::Sqrt(pow(eCmeas/Cmeas,2)+pow(eRmeas/Rmeas,2));

  cout<<"R = ("<<Rmeas<<" +/- "<<eRmeas<<") Ohm"<<endl;
  cout<<"C = ("<<Cmeas<<" +/- "<<eCmeas<<") F"<<endl;
  cout<<"f_t expected = ("<<ftexp<<" +/- "<<eftexp<<") Hz"<<endl;

  //======================================== test senza circuito attaccato
  
  TCanvas *c0v = new TCanvas("c0v","c0v",1200,1200);
  c0v->Divide(1,2);
  c0v->GetPad(1)->SetLogx();
  c0v->GetPad(2)->SetLogx();
  
  ifstream fvu("risposta_a_vuoto.txt");// scala 1 V
  double Vsonda1ch1[N]={0};
  double Vsonda1ch2[N]={0};
  double Vsonda2ch1[N]={0};
  double Vsonda2ch2[N]={0};
  double eVsonda1ch1[N]={0};
  double eVsonda1ch2[N]={0};
  double eVsonda2ch1[N]={0};
  double eVsonda2ch2[N]={0};
  double Ts[N]={0};
  double frs[N]={0};
  double efrs[N]={0};
  
  for(int i=0; i<N; i++){
    fvu>>Ts[i]>>Vsonda1ch1[i]>>Vsonda1ch2[i]>>Vsonda2ch1[i]>>Vsonda2ch2[i];
    frs[i]=1./Ts[i];
    eVsonda1ch1[i]=(1.5/100.)*Vsonda1ch1[i];
    eVsonda2ch1[i]=(1.5/100.)*Vsonda2ch1[i];
    eVsonda1ch2[i]=(1.5/100.)*Vsonda1ch2[i];
    eVsonda2ch2[i]=(1.5/100.)*Vsonda2ch2[i];	
  }

  //TGraphErrors* gs1ch1 = new TGraphErrors(N,frs,Vsonda1ch1,efrs,eVsonda1ch1);
  //TGraphErrors* gs1ch2 = new TGraphErrors(N,frs,Vsonda1ch2,efrs,eVsonda1ch2);
  //TGraphErrors* gs2ch1 = new TGraphErrors(N,frs,Vsonda2ch1,efrs,eVsonda2ch1);
  //TGraphErrors* gs2ch2 = new TGraphErrors(N,frs,Vsonda2ch2,efrs,eVsonda2ch2);

  TGraph* gs1ch1 = new TGraph(N,frs,Vsonda1ch1);
  TGraph* gs1ch2 = new TGraph(N,frs,Vsonda1ch2);
  TGraph* gs2ch1 = new TGraph(N,frs,Vsonda2ch1);
  TGraph* gs2ch2 = new TGraph(N,frs,Vsonda2ch2);

  gs1ch1->SetMarkerStyle(20);
  gs1ch2->SetMarkerStyle(20);
  gs2ch1->SetMarkerStyle(24);
  gs2ch2->SetMarkerStyle(24);

  gs1ch1->SetMarkerColor(2);
  gs1ch2->SetMarkerColor(4);
  gs1ch1->SetLineColor(2);
  gs1ch2->SetLineColor(4);

  gs2ch1->SetMarkerColor(2);
  gs2ch2->SetMarkerColor(4);
  gs2ch1->SetLineColor(2);
  gs2ch2->SetLineColor(4);

  TH2F* hfr0 = new TH2F("hfr0","risposta a vuoto",100,0.9e-4,0.3,100,7.3,8);
  hfr0->SetStats(kFALSE);

  c0v->cd(1);
  hfr0->Draw();
  gs1ch1->Draw("PLSAME");
  gs1ch2->Draw("PLSAME");
  gs2ch1->Draw("PLSAME");
  gs2ch2->Draw("PLSAME");

  TLegend* mylege = new TLegend(0.7,0.7,0.99,0.99);
  mylege->AddEntry(gs1ch1,"sonda 1 ch 1","LP");
  mylege->AddEntry(gs2ch1,"sonda 2 ch 1","LP");
  mylege->AddEntry(gs1ch2,"sonda 1 ch 2","LP");  
  mylege->AddEntry(gs2ch2,"sonda 2 ch 2","LP");

  mylege->SetLineColor(0);
  mylege->SetLineWidth(0);
  mylege->Draw();
  
  double weight_s1ch1[N]={1.};
  double weight_s1ch2[N]={1.};

  double weight_s2ch1[N]={1.};
  double weight_s2ch2[N]={1.};

  double pn=Vsonda2ch2[0];
  for(int i=0; i<N; i++){
    weight_s1ch1[i]=Vsonda1ch1[i]/pn;
    weight_s1ch2[i]=Vsonda1ch2[i]/pn;
    weight_s2ch1[i]=Vsonda2ch1[i]/pn;
    weight_s2ch2[i]=Vsonda2ch2[i]/pn;
  }

  TGraph* gcs1ch1 = new TGraph(N,frs,weight_s1ch1);
  TGraph* gcs1ch2 = new TGraph(N,frs,weight_s1ch2);
  TGraph* gcs2ch1 = new TGraph(N,frs,weight_s2ch1);
  TGraph* gcs2ch2 = new TGraph(N,frs,weight_s2ch2);
  
  c0v->cd(2);

  gcs1ch1->SetMarkerStyle(20);
  gcs1ch2->SetMarkerStyle(20);
  gcs2ch1->SetMarkerStyle(24);
  gcs2ch2->SetMarkerStyle(24);

  gcs1ch1->SetMarkerColor(2);
  gcs1ch2->SetMarkerColor(4);
  gcs1ch1->SetLineColor(2);
  gcs1ch2->SetLineColor(4);

  gcs2ch1->SetMarkerColor(2);
  gcs2ch2->SetMarkerColor(4);
  gcs2ch1->SetLineColor(2);
  gcs2ch2->SetLineColor(4);
  
  TH2F* hfr1 = new TH2F("hfr1","ripesamento",100,0.9e-4,0.3,100,0.95,1.01);
  hfr1->SetStats(kFALSE);

  hfr1->Draw();
  gcs1ch1->Draw("PLSAME");
  //gcs1ch2->Draw("PLSAME");
  //gcs2ch1->Draw("PLSAME");
  gcs2ch2->Draw("PLSAME");

  c0v->SaveAs("RCF_vuoto.pdf");
  c0v->SaveAs("RCF_vuoto.png");

  //=================================


  
  ifstream f(infil);
  
  double T[N]={0};
  double DT[N]={0};
  double eDT[N]={0};
  double V[N]={0};
  double Vin[N]={0};
  double Vinnorm[N]={0};
  double eVinnorm[N]={0};
  double phi[N]={0};
  double fr[N]={0};
  double logfr[N]={0};
  double fsq[N]={0};
  double ifsq[N]={0};
  double A[N]={0};
  double logA[N]={0};
  double Amd[N]={0};
  
  double phin[N]={0};
  
  double tanphi[N]={0};
  double cotanphi[N]={0};
  
  double eT[N]={0};
  double eV[N]={0};
  double ephi[N]={0};
  double efr[N]={0};
  double elogfr[N]={0};
  double efsq[N]={0};
  double eifsq[N]={0};
  double eA[N]={0};
  double elogA[N]={0};
  double eAmd[N]={0};
  
  double ephin[N]={0};

  double etanphi[N]={0};
  double ecotanphi[N]={0};

  double vscal[N]={0};
  double tscal[N]={0};
  double tscalDT[N]={0};
  
  double div=1;// 1 V/div
  double tdiv=1;//mus diverso per ogni punto!!!!

  double xmin=0;
  double xmax=0;
  //if(opt==1){
  xmin=1e-1;
  xmax=20e3;
  //}
  //if(opt==2){
  //xmin=5e-1;
  //xmax=2e2;
  //}
  double dummy=0;
  double eV0=div*0.04;//V
  int npo=0;
  cout<<" T (us), V_in (V), V_out (V), Delta t (us), V/div, us/div"<<endl;

  for (int i=0; i<N; i++){
    f>>T[i]>>Vin[i]>>V[i]>>vscal[i]>>DT[i]>>tscalDT[i];
    fr[i]=1e-3/(1e-6*T[i]);
    //fr[i]*=0.8;
    logfr[i]=TMath::Log10(fr[i]);
    fsq[i]=pow(fr[i],2);
    ifsq[i]=pow(1./fr[i],2);

    eDT[i]=0.04*tscalDT[i]*TMath::Sqrt(2.);
    phi[i]=360.*DT[i]/T[i];
    ephi[i]=phi[i]*eDT[i]/DT[i];

    //cout<<" T "<<T[i]<<" us, f = "<<fr[i]<<" kHz, phi = "<<phi[i]<<" A = "<<A[i]<<endl;
    cout<<T[i]
	<<"\t"<<Vin[i]
      	<<"\t"<<V[i]
	<<"\t"<<DT[i]	
	<<"\t"<<vscal[i]
	<<"\t"<<tscalDT[i]
	<<endl;
    
    
    double DR=TMath::Pi()/180.;
    tanphi[i]=TMath::Tan(phi[i]*DR);
    cotanphi[i]=1./tanphi[i];
    etanphi[i]=pow(1./TMath::Cos(phi[i]*DR),2)*(ephi[i]*DR);
    ecotanphi[i]=pow(1./TMath::Sin(phi[i]*DR),2)*(ephi[i]*DR);
    
    phin[i]=phi[i]/90.;
    ephin[i]=ephi[i]/90.;
    
    efr[i]=0;
    elogfr[i]=0;

    efsq[i]=2*fr[i]*efr[i];
    eifsq[i]=(2./pow(fr[i],3))*efr[i];
    if(T[i]!=0)npo++;
  }
  f.close();

  bool enableREW = false;
  
  for(int i=0; i<npo; i++){
    
    //if(opt==1)A[i]=(V[i]/Vin[i])/1.02632;// riporto a mano a 1!!!
    if(opt==1)A[i]=(V[i]/Vin[i])/1.;// riporto a mano a 1!!!
    if(opt==2)A[i]=(V[i]/Vin[i])/1.;// riporto a mano a 1!!!
    
    A[i]=(V[i]/Vin[i]);
    if(enableREW)A[i]=(weight_s2ch2[i]*V[i])/(weight_s1ch1[i]*Vin[i]);
    //if(opt==1)A[i]=(V[i]/V[0]);
    //if(opt==2)A[i]=(V[i]/V[N-1]);

    Amd[i]=1./pow(A[i],2);
    double cambioscala=1;
    //if(vscal[i]==1)cambioscala=0; sempre preente errore di cambio scala perche' usati due canali diversi
    eA[i]=A[i]*TMath::Sqrt(pow(0.04/Vin[i],2)+pow((vscal[i]*0.04)/V[i],2)+cambioscala*(2*pow(0.015,2)));
    logA[i]=TMath::Log10(A[i]);
    elogA[i]=eA[i]/A[i];
    
    //if(cambioscala)A[i]*=1.25;    
    // A = ki*Vout/k0*Vin
    
    eAmd[i]=(2./pow(A[i],3))*eA[i];    
     
    Vinnorm[i]=Vin[i]/Vin[0];
    eVinnorm[i]=0.01;// sistemare!!!!!!


  }
  
  //========================================

  
  TGraphErrors* g = new TGraphErrors(npo,fr,A,efr,eA);
  g->SetTitle("");
  g->GetXaxis()->SetTitle("f (kHz)");
  g->GetYaxis()->SetTitle("A = V_{out} / V_{in}");

  TGraphErrors* gVin = new TGraphErrors(npo,fr,Vinnorm,efr,eVinnorm);
  gVin->SetTitle("");
  gVin->GetXaxis()->SetTitle("f (kHz)");
  gVin->GetYaxis()->SetTitle("V_{in}/V_{in}(0)");
  gVin->SetMarkerStyle(25);
  
  TGraphErrors* glog = new TGraphErrors(npo,logfr,logA,elogfr,elogA);
  glog->SetTitle("");
  glog->GetXaxis()->SetTitle("log f (kHz)");
  glog->GetYaxis()->SetTitle("log A = log (V_{out} / V_{in})");

  TGraphErrors* gphi = new TGraphErrors(npo,fr,phin,efr,ephin);
  gphi->SetTitle("");
  gphi->GetXaxis()->SetTitle("f (kHz)");
  gphi->GetYaxis()->SetTitle("#phi (deg.)");
  
  //===============================================================
  TCanvas *c2 = new TCanvas("c2","c2",1200,1200);
  c2->Divide(1,2);
  c2->cd(1);
  c2->GetPad(1)->SetLogx();
  c2->GetPad(2)->SetLogx();

  c2->GetPad(1)->SetLogy();

  TGraphErrors* gFA;
  if(opt==1)gFA = new TGraphErrors(npo,fsq,Amd,efsq,eAmd);
  if(opt==2)gFA = new TGraphErrors(npo,ifsq,Amd,eifsq,eAmd);
  gFA->SetTitle("");
  gFA->SetMinimum(0.1);
  gFA->SetMarkerStyle(20);
  if(opt==1)gFA->GetXaxis()->SetTitle("f^{2} (kHz^{2})");
  if(opt==2)gFA->GetXaxis()->SetTitle("f^{-2} (kHz^{-2})");
  gFA->GetYaxis()->SetTitle("1/A^{2}");
  gFA->Fit("pol1");
  gFA->Draw("AP");

  double m=gFA->GetFunction("pol1")->GetParameter(1);
  double em=gFA->GetFunction("pol1")->GetParError(1);
  double q=gFA->GetFunction("pol1")->GetParameter(0);
  double eq=gFA->GetFunction("pol1")->GetParError(0);
  double RR=gFA->GetCorrelationFactor();

  double res[N]={0};
  double eres[N]={0};
  TF1* fit = (TF1*)gFA->GetFunction("pol1");
  double mysum=0;
  double chi2=0;
  for(int i=0; i<npo; i++){
    eres[i]=eAmd[i];
    res[i]=Amd[i]-fit->Eval(fsq[i]);
    if(eres[i]>0){
      chi2+=pow(res[i]/eres[i],2);
    }
    mysum+=pow(res[i],2);
  }
  double sigmay_pos=TMath::Sqrt(mysum/((double)npo-2.));

  double ftmeas=1./TMath::Sqrt(m);
  double eftmeas=0.5*ftmeas*(em/m);

  if(opt==2){
    ftmeas=TMath::Sqrt(m);
    eftmeas=0.5*ftmeas*(em/m);
  }
 
  if(opt==1)cout<<"============== A^-2 vs f^2 ===================="<<endl;
  if(opt==2)cout<<"============== A^-2 vs f^-2 ===================="<<endl;
  cout <<" m = ("<<m<<" +/- "<<em<<"), err. rel. = "<<100*em/m <<" %"<<endl;
  cout <<" q = ("<<q<<" +/- "<<eq<<") "<<endl;
  cout <<" R = "<<RR<<endl;
  cout <<" sigma_y pos = "<<sigmay_pos<<" mV"<<endl;
  cout <<" chi2 = "<<chi2<<"/"<<(npo-2)<<endl;
  cout <<" f_t = ("<<ftmeas<<" +/- "<<eftmeas<<")"<<endl;
  cout<<"======================================="<<endl;

  double xtext=2e-4;
  if(opt==1)xtext=2e-2;
  if(opt==2)xtext=2e-4;
 
  TLatex *tGA;
  if(opt==1)tGA = new TLatex(xtext,1e2,Form("#frac{1}{A^{2}} = (%1.4f #pm %1.4f) f^{2} + (%1.4f #pm %1.4f)",m,em,q,eq));
  if(opt==2)tGA = new TLatex(xtext,1e2,Form("#frac{1}{A^{2}} = #frac{(%1.4f #pm %1.4f)}{f^{2}} + (%1.4f #pm %1.4f)",m,em,q,eq));
  tGA->Draw();
  TLatex *tGA1 = new TLatex(xtext,1e1,Form("f_{t} = (%1.3f #pm %1.3f) kHz",ftmeas,eftmeas));
  tGA1->Draw();
  
  c2->cd(2);
  TF1* ff=(TF1*)gFA->GetFunction("pol1");
  double residuU[N]={0};
  double eresiduU[N]={0};
  for(int i=0; i<npo; i++){
    eresiduU[i]=eAmd[i];
    if(opt==1)residuU[i]=Amd[i]-ff->Eval(fsq[i]);
    if(opt==2)residuU[i]=Amd[i]-ff->Eval(ifsq[i]);
  }
  TGraphErrors* gresiU;
  if(opt==1)gresiU = new TGraphErrors(npo,fsq,residuU,efsq,eresiduU);
  if(opt==2)gresiU = new TGraphErrors(npo,ifsq,residuU,eifsq,eresiduU);
  gresiU->SetTitle("");
  if(opt==1){
    gresiU->GetXaxis()->SetTitle("f^{2} (kHz^{2})");
    gresiU->GetYaxis()->SetTitle("A^{-2} (meas.-fit)");
  }
  if(opt==2){
    gresiU->GetXaxis()->SetTitle("f^{-2} (kHz^{-2})");
    gresiU->GetYaxis()->SetTitle("A^{-2} (meas.-fit)");
  }
 
  gresiU->SetMarkerStyle(20);
  //gresiU->GetXaxis()->SetRangeUser(xmin,xmax);
  gresiU->GetYaxis()->SetRangeUser(-0.3,0.3);
  gresiU->Draw("AP");
  TLine *linU;
  if(opt==1)linU = new TLine(1e-2,0,1e4,0);
  if(opt==2)linU = new TLine(1e-4,0,1e2,0);
  linU->SetLineStyle(2);
  linU->Draw();
 
  c2->SaveAs(Form("RCF_0_%s.pdf",lab));
  c2->SaveAs(Form("RCF_0_%s.png",lab));

  TCanvas *c3 = new TCanvas("c3","c3",1200,1200);
  c3->Divide(1,2);
  c3->cd(1);
  //c3->GetPad(1)->SetLogx();
  //c3->GetPad(1)->SetLogy();

  TGraphErrors* gtgf;
  if(opt==1)gtgf= new TGraphErrors(npo,fr,tanphi,efr,etanphi);
  if(opt==2)gtgf= new TGraphErrors(npo,fr,cotanphi,efr,ecotanphi);
  gtgf->SetTitle("");
  gtgf->SetMinimum(0.1);
  gtgf->SetMarkerStyle(20);
  gtgf->GetXaxis()->SetTitle("f (kHz)");
  if(opt==1)gtgf->GetYaxis()->SetTitle("tan #phi");
  if(opt==2)gtgf->GetYaxis()->SetTitle("cotan #phi");
  double frmax=2.1;
  if(opt2==-13)frmax*=100;
  TF1 *fit1 = new TF1("fit1","pol1",0,frmax);
  gtgf->Fit("fit1","R");
  gtgf->GetXaxis()->SetRangeUser(0,frmax);
  gtgf->GetYaxis()->SetRangeUser(0,1);
  gtgf->Draw("AP");
  //gtgf->Print();
 
  m=gtgf->GetFunction("fit1")->GetParameter(1);
  em=gtgf->GetFunction("fit1")->GetParError(1);
  q=gtgf->GetFunction("fit1")->GetParameter(0);
  eq=gtgf->GetFunction("fit1")->GetParError(0);
  RR=gtgf->GetCorrelationFactor();

  mysum=0;
  chi2=0;
  for(int i=0; i<N; i++){
      if(opt==1){
      eres[i]=etanphi[i];
      res[i]=tanphi[i]-fit1->Eval(fr[i]);
    }
    if(opt==2){
      eres[i]=ecotanphi[i];
      res[i]=cotanphi[i]-fit1->Eval(fr[i]);
    }
    if(eres[i]>0){
      chi2+=pow(res[i]/eres[i],2);
      //cout<<"tan(f): meas-fit "<<i<<" "<<res[i]<<" "<<eres[i]<<" "<<pow(res[i]/eres[i],2)<<endl;
    }
    mysum+=pow(res[i],2);
  }
  sigmay_pos=TMath::Sqrt(mysum/((double)npo-2.));

  double ftmeas1=1./m;
  double eftmeas1=ftmeas1*(em/m);
 
  if(opt==1)cout<<"============== tan(phi) vs f ===================="<<endl;
  if(opt==2)cout<<"============== cotan(phi) vs f ===================="<<endl;
  cout <<"| m = ("<<m<<" +/- "<<em<<"), err. rel. = "<<100*em/m <<" %"<<endl;
  cout <<"| q = ("<<q<<" +/- "<<eq<<") "<<endl;
  cout <<"| R = "<<RR<<endl;
  cout <<"| sigma_y pos = "<<sigmay_pos<<" mV"<<endl;
  cout <<"| chi2 = "<<chi2<<"/"<<(npo-2)<<endl;
  cout <<"| f_t = ("<<ftmeas1<<" +/- "<<eftmeas1<<")"<<endl;
  cout<<"================================================="<<endl;


  TLatex *tphi = new TLatex(1.0,0.4,Form("tan #phi = (%1.4f #pm %1.4f) f + (%1.4f #pm %1.4f)",m,em,q,eq));
  tphi->Draw();
  TLatex *tphi1 = new TLatex(1.0,0.3,Form("f_{t} = (%1.3f #pm %1.3f) kHz",ftmeas1,eftmeas1));
  tphi1->Draw();
  
  c3->cd(2);

  double residuO[N]={0};
  double eresiduO[N]={0};
  for(int i=0; i<npo; i++){
    if(opt==1)eresiduO[i]=etanphi[i];
    if(opt==2)eresiduO[i]=ecotanphi[i];
    if(opt==1)residuO[i]=tanphi[i]-fit1->Eval(fr[i]);
    if(opt==2)residuO[i]=cotanphi[i]-fit1->Eval(fr[i]);
  }
  TGraphErrors* gresiO = new TGraphErrors(npo,fr,residuO,efr,eresiduO);
  gresiO->SetTitle("");
  gresiO->GetXaxis()->SetTitle("f (kHz)");
  gresiO->GetYaxis()->SetTitle("tan#phi (meas.-fit)");
  gresiO->SetMarkerStyle(20);
  gresiO->GetXaxis()->SetRangeUser(0,frmax);
  double rrange=0.03;
  if(opt==2)rrange=0.1;
  gresiO->GetYaxis()->SetRangeUser(-rrange,rrange);
  gresiO->Draw("AP");
  TLine *linO = new TLine(0,0,frmax,0);
  linO->SetLineStyle(2);
  linO->Draw();

 
  c3->SaveAs(Form("RCF_1_%s.pdf",lab));
  c3->SaveAs(Form("RCF_1_%s.png",lab));

  //===========================================================================
  const int NSTEP=50;

  double fftmin=ftexp*0.9*1e-3;
  double fftmax=ftexp*1.1*1e-3;

  double ou[NSTEP]={0};
  double ouf[NSTEP]={0};
  //double ouNR[NSTEP]={0};
  //double oufNR[NSTEP]={0};
  double fft[NSTEP]={0};

  double mchi2min0=1e9;
  double mchi2min1=1e9;
  double fmin0=0;
  double fmin1=0;
 
  for(int i=0; i<NSTEP; i++){
    fft[i]=fftmin+i*(fftmax-fftmin)/(double)NSTEP;

    //ou[i]=GetCHIS(fft[i],g,opt,1)/((double)N-1.);
    //ouf[i]=GetCHIS(fft[i],gphi,opt,2)/((double)N-1.);

    ou[i]=GetCHIS(fft[i],g,opt,1);
    ouf[i]=GetCHIS(fft[i],gphi,opt,2);
    if(ou[i]<=mchi2min0){
      mchi2min0=ou[i];
      fmin0=fft[i];
    }
    if(ouf[i]<=mchi2min1){
      mchi2min1=ouf[i];
      fmin1=fft[i];
    }
    //ouNR[i]=ou[i]*((double)N-1.);
    //oufNR[i]=ouf[i]*((double)N-1.);

    //cout<<i<<" "<<fft[i]<<" "<<ou<<endl;
  }


 
 
  TGraphErrors* gchi2 = new TGraphErrors(NSTEP,fft,ou);
  TGraphErrors* gfchi2 = new TGraphErrors(NSTEP,fft,ouf);
 
  TCanvas *c5 = new TCanvas("c5","c5",1200,800);
  c5->Divide(2,1);
  c5->cd(1);
  gchi2->SetMarkerStyle(20);
  gchi2->SetTitle("#chi^{2}(f_{t}) per A");
  gchi2->GetXaxis()->SetTitle("f_{t} (kHz)");
  gchi2->GetYaxis()->SetTitle("#chi^{2}");

  fftmin=0.95*fmin0;
  fftmax=1.05*fmin0;
 
  TF1 *fitq = new TF1("fitq","pol2",fftmin,fftmax);
  gchi2->Fit("fitq","R");
  gchi2->Draw("AP");
  TF1* g2 = gchi2->GetFunction("fitq");
  double c=g2->GetParameter(0);
  double b=g2->GetParameter(1);
  double a=g2->GetParameter(2);
  double ftfit=-b/(2.*a);
  double min=a*pow(ftfit,2)+b*ftfit+c;
  cout<<"ftfit "<<ftfit<<" min "<<min<<endl;
  TLine *ll9 = new TLine(ftfit,0,ftfit,10);
  TLine *ll10 = new TLine(fftmin,min,fftmax,min);
  TLine *ll11 = new TLine(fftmin,min+1,fftmax,min+1);
  ll9->Draw();
  ll10->Draw();
  ll11->Draw();
  double cc1=c-min-1.;
  double x12=(-b+TMath::Sqrt(pow(b,2)-4*a*cc1))/(2.*a);
  double x21=(-b-TMath::Sqrt(pow(b,2)-4*a*cc1))/(2.*a);
  TLine *ll12 = new TLine(x12,0,x12,10);
  TLine *ll21 = new TLine(x21,0,x21,10);
 
  ll12->Draw();
  ll21->Draw();
  double eftfit=x12-ftfit;
 
  c5->cd(2);
  gfchi2->SetMarkerStyle(20);
  gfchi2->SetTitle("#chi^{2}(f_{t}) per  D#phi");
  gfchi2->GetXaxis()->SetTitle("f_{t} (kHz)");
  gfchi2->GetYaxis()->SetTitle("#chi^{2}");

  fftmin=0.95*fmin1;
  fftmax=1.05*fmin1;
 
  TF1 *fitqf = new TF1("fitqf","pol2",fftmin,fftmax);
  gfchi2->Fit("fitqf","R");
  gfchi2->Draw("AP");
  TF1* gf2 = gfchi2->GetFunction("fitqf");
  double cf=gf2->GetParameter(0);
  double bf=gf2->GetParameter(1);
  double af=gf2->GetParameter(2);
  double ftfitf=-bf/(2.*af);
  double minf=af*pow(ftfitf,2)+bf*ftfitf+cf;
  cout<<"ftfitf "<<ftfitf<<" minf "<<minf<<endl;
  TLine *llf9 = new TLine(ftfitf,0,ftfitf,10);
  TLine *llf10 = new TLine(fftmin,minf,fftmax,minf);
  TLine *llf11 = new TLine(fftmin,minf+1,fftmax,minf+1);
  llf9->Draw();
  llf10->Draw();
  llf11->Draw();
  double ccf1=cf-minf-1.;
  double xf12=(-bf+TMath::Sqrt(pow(bf,2)-4*af*ccf1))/(2.*af);
  double xf21=(-bf-TMath::Sqrt(pow(bf,2)-4*af*ccf1))/(2.*af);
  TLine *llf12 = new TLine(xf12,0,xf12,10);
  TLine *llf21 = new TLine(xf21,0,xf21,10);
  llf12->Draw();
  llf21->Draw();
  double eftfitf=xf12-ftfitf;

  c5->SaveAs(Form("RCF_2_%s.pdf",lab));
  c5->SaveAs(Form("RCF_2_%s.png",lab)); 

  //===========================================================================
  TCanvas *c0 = new TCanvas("c0","c0",1200,1200);
  c0->Divide(1,1);
  c0->cd(1);
  c0->GetPad(1)->SetLogx();

  g->SetMarkerStyle(20);
  g->SetMarkerColor(kBlack);
  g->SetLineColor(kBlack);

  gphi->SetMarkerStyle(24);
  gphi->SetMarkerColor(kRed);
  gphi->SetLineColor(kRed);
 
  g->Draw("AP");
  gphi->Draw("PSAME");
  gVin->Draw("PSAME");

  TGaxis *axis1 = new TGaxis(1.1*fr[npo-1],0,1.1*fr[npo-1],1,0,1,510,"+L");
  axis1->SetLineColor(kRed);
  axis1->SetLabelColor(kRed);
  axis1->SetTitle("D#phi (^{#circ}/90)");
  axis1->SetTextColor(kRed);
  axis1->SetLabelSize(0.03);
  axis1->SetTextSize(0.03);
  axis1->Draw();

  TF1* func0;
  if(opt==1)func0 = new TF1("func0",Form("1/TMath::Sqrt(1+(x/%f)^2)",ftfit),xmin,xmax);
  if(opt==2)func0 = new TF1("func0",Form("1/TMath::Sqrt(1+(%f/x)^2)",ftfit),xmin,xmax);
  func0->SetLineColor(kBlack);
  func0->SetLineStyle(3);
  func0->Draw("same");
  TF1* func00;
  if(opt==1)func00 = new TF1("func00",Form("1/TMath::Sqrt(1+(x/%f)^2)",ftfit),xmin,xmax);
  if(opt==2)func00 = new TF1("func00",Form("1/TMath::Sqrt(1+(%f/x)^2)",ftfit),xmin,xmax);
  func00->SetLineColor(kBlack);
  func00->SetLineStyle(3);

  TF1* func1;
  //ftfitf=ftexp/1e3;//tmp!!
  if(opt==1)func1 = new TF1("func1",Form("(2./TMath::Pi())*TMath::ATan(x/%f)",ftfitf),xmin,xmax);
  if(opt==2)func1 = new TF1("func1",Form("(2./TMath::Pi())*TMath::ATan(%f/x)",ftfitf),xmin,xmax);
  func1->SetLineColor(kRed);
  func1->SetLineStyle(2);
  func1->Draw("same");
  TF1* func11;
  if(opt==1)func11 = new TF1("func11",Form("(2./TMath::Pi())*TMath::ATan(x/%f)",ftmeas1),xmin,xmax);
  if(opt==2)func11 = new TF1("func11",Form("(2./TMath::Pi())*TMath::ATan(%f/x)",ftmeas1),xmin,xmax);
  func11->SetLineColor(kRed);
  func11->SetLineStyle(2);

  TLine *ll = new TLine(ftmeas,0,ftmeas,1);
  ll->SetLineStyle(3);
  ll->Draw();

  TLine *ll1 = new TLine(xmin,1,xmax,1);
  ll1->SetLineStyle(2);
  ll1->Draw();

  TLine *ll2 = new TLine(xmin,1./TMath::Sqrt(2.),ftmeas,1./TMath::Sqrt(2.));
  ll2->SetLineStyle(2);
  ll2->SetLineColor(kBlack);
  ll2->Draw();

  TLine *ll3 = new TLine(ftmeas,0.5,xmax,0.5);
  ll3->SetLineStyle(2);
  ll3->SetLineColor(kRed);
  ll3->Draw();

  /*

  ifstream fqucs0;
  ifstream fqucs1;

  ifstream fqucs2;
  ifstream fqucs3;

  ifstream fqucs4;
  ifstream fqucs5;

  fqucs0.open("/home/longhin/work/didattica/Sperimentazioni2Fisica/2021/simulazioni_qucs/passa_basso/T_PB.txt");
  fqucs1.open("/home/longhin/work/didattica/Sperimentazioni2Fisica/2021/simulazioni_qucs/passa_basso/T_PA.txt");
  fqucs2.open("/home/longhin/work/didattica/Sperimentazioni2Fisica/2021/simulazioni_qucs/passa_basso/Vin_PB.txt");
  fqucs3.open("/home/longhin/work/didattica/Sperimentazioni2Fisica/2021/simulazioni_qucs/passa_basso/Vin_PA.txt");
  fqucs4.open("/home/longhin/work/didattica/Sperimentazioni2Fisica/2021/simulazioni_qucs/passa_basso/sfas_PB.txt");
  fqucs5.open("/home/longhin/work/didattica/Sperimentazioni2Fisica/2021/simulazioni_qucs/passa_basso/sfas_PA.txt");
  
  const int NQ=300;
  double frqucs[NQ]={0};

  double TPBq[NQ]={0};
  double TPAq[NQ]={0};
  double VinqPA[NQ]={0};
  double VinqPB[NQ]={0};
  double sfaPBq[NQ]={0};
  double sfaPAq[NQ]={0};

  double reale=0;
  double immaginaria=0;
  dummy=0;

  for (int i=0; i<300; i++) {
    fqucs0>>frqucs[i]>>reale>>immaginaria;
    TPBq[i]=TMath::Sqrt(pow(reale,2)+pow(immaginaria,2));
    frqucs[i]*=1e-3;
  }
  for (int i=0; i<300; i++) {
    fqucs1>>frqucs[i]>>reale>>immaginaria;
    TPAq[i]=TMath::Sqrt(pow(reale,2)+pow(immaginaria,2));
    frqucs[i]*=1e-3;
  }
  for (int i=0; i<300; i++) {
    fqucs2>>frqucs[i]>>VinqPB[i]>>dummy;
    frqucs[i]*=1e-3;
  }
  for (int i=0; i<300; i++) {
    fqucs3>>frqucs[i]>>VinqPA[i]>>dummy;
    frqucs[i]*=1e-3;
  }
  for (int i=0; i<300; i++) {
    fqucs4>>frqucs[i]>>sfaPBq[i]>>dummy;
    sfaPBq[i]/=90.;
    frqucs[i]*=1e-3;
  }
  for (int i=0; i<300; i++) {
    fqucs5>>frqucs[i]>>sfaPAq[i]>>dummy;
    //cout <<sfaPAq[i]<<endl;
    sfaPAq[i]/=90.;
    frqucs[i]*=1e-3;
  }
  
  TGraph* gqucs0 = new TGraph(NQ,frqucs,TPBq);
  gqucs0->SetLineColor(kMagenta);
  gqucs0->SetMarkerColor(kMagenta);
  if(opt==1)gqucs0->Draw("LSAME");

  TGraph* gqucs1 = new TGraph(NQ,frqucs,TPAq);
  gqucs1->SetLineColor(kMagenta+2);
  gqucs1->SetMarkerColor(kMagenta+2);
  if(opt==2)gqucs1->Draw("LSAME");

  TGraph* gqucs2 = new TGraph(NQ,frqucs,VinqPB);
  gqucs2->SetLineColor(kOrange);
  gqucs2->SetMarkerColor(kOrange);
  if(opt==1)gqucs2->Draw("LSAME");

  TGraph* gqucs3 = new TGraph(NQ,frqucs,VinqPA);
  gqucs3->SetLineColor(kOrange+2);
  gqucs3->SetMarkerColor(kOrange+2);
  if(opt==2)gqucs3->Draw("LSAME");

  TGraph* gqucs4 = new TGraph(NQ,frqucs,sfaPBq);
  gqucs4->SetLineColor(kGreen);
  gqucs4->SetMarkerColor(kGreen);
  if(opt==1)gqucs4->Draw("LSAME");

  TGraph* gqucs5 = new TGraph(NQ,frqucs,sfaPAq);
  gqucs5->SetLineColor(kGreen+2);
  gqucs5->SetMarkerColor(kGreen+2);
  if(opt==2)gqucs5->Draw("LSAME");

  c0->SaveAs(Form("RCF_3_%s.pdf",lab));
  c0->SaveAs(Form("RCF_3_%s.png",lab)); 


  TCanvas *c0q = new TCanvas("c0q","c0q",1200,1200);
  c0q->Divide(1,2);
  c0q->GetPad(1)->SetLogx();
  c0q->GetPad(2)->SetLogx();
  //c0q->GetPad(3)->SetLogx();

  double ftqucs=(1e-3/(2*TMath::Pi()*1.204e3*96e-9));
  TLine *lft = new TLine(ftqucs,0,ftqucs,1);
  lft->SetLineStyle(2);
  double vall=1./TMath::Sqrt(2.);
  TLine *lft1 = new TLine(1,vall,2,vall);
  lft1->SetLineStyle(2);
  TLine *lft2 = new TLine(1,0.5,2,0.5);
  lft2->SetLineStyle(2);

  gqucs0->SetTitle("QUCS T");
  gqucs1->SetTitle("QUCS T");

  gqucs2->SetTitle("QUCS V_{in}");
  gqucs3->SetTitle("QUCS V_{in}");
  
  gqucs4->SetTitle("QUCS phase");
  gqucs5->SetTitle("QUCS phase");
  
  c0q->cd(1);
  gqucs0->Draw("AL");
  gqucs1->Draw("LSAME");
  lft->Draw();
  lft1->Draw();
  //c0q->cd(2);
  //gqucs2->Draw("AL");
  //gqucs3->Draw("LSAME");
  //lft->Draw();
  c0q->cd(2);
  gqucs4->Draw("AL");
  gqucs5->Draw("LSAME");
  lft2->Draw();
  lft->Draw();
  c0q->SaveAs("RCF_qucs.pdf");
  c0q->SaveAs("RCF_qucs.png"); 

  */
  
  //===========================================================================
  TCanvas *c00 = new TCanvas("c00","c00",1200,1200);
  c00->Divide(1,1);
  c00->cd(1);
  
  TH2F* hfr = new TH2F("hfr","",100,fl0,fl1,100,amin,amax);
  hfr->SetStats(kFALSE);
  hfr->GetXaxis()->SetTitle("f (kHz)");
  hfr->GetYaxis()->SetTitle("A");
  hfr->Draw();


  TFitResultPtr r = g->Fit("pol1","S","",fitfl0,fitfl1);
  TMatrixD cov = r->GetCorrelationMatrix();
  cov.Print();

  TF1* flin0 =(TF1*)g->GetFunction("pol1");
  double par0=flin0->GetParameter(0);
  double par1=flin0->GetParameter(1);
  double epar0=flin0->GetParError(0);
  double epar1=flin0->GetParError(1);
  double y0=1./TMath::Sqrt(2.);
  double ftlin=(y0-par0)/par1;
  cout<<"freq taglio lineare intorno su A "<<ftlin<<" "<<par0<<" "<<par1<<endl;
  double eftlin=ftlin*TMath::Sqrt(pow(epar0/(y0-par0),2)+pow(epar1/par1,2)-2*0.9977*(TMath::Abs(epar0/(y0-par0)))*(TMath::Abs(epar1/par1)));


  TFitResultPtr rf = gphi->Fit("pol1","S","",fitfl0,fitfl1);
  TMatrixD covf = rf->GetCorrelationMatrix();
  covf.Print();
  TF1* flin1 =(TF1*)gphi->GetFunction("pol1");
  double parf0=flin1->GetParameter(0);
  double parf1=flin1->GetParameter(1);
  double eparf0=flin1->GetParError(0);
  double eparf1=flin1->GetParError(1);
  //y0=0.5;
  double RG=0;
  //if(opt==1)RG=50;
  y0=0.5*(Rmeas+RG)/Rmeas;
 
  double ftlinf=(y0-parf0)/parf1;
  double eftlinf=ftlinf*TMath::Sqrt(pow(eparf0/(y0-parf0),2)+pow(eparf1/parf1,2)-2*0.9977*(TMath::Abs(eparf0/(y0-parf0)))*(TMath::Abs(eparf1/parf1)));

  cout << "\n===== FIT LINEARE LOCALE SU A(f) =====" << endl;
  cout << "intervallo = [" << fitfl0 << ", " << fitfl1 << "] kHz" << endl;
  cout << "par0 = " << par0 << " +/- " << epar0 << endl;
  cout << "par1 = " << par1 << " +/- " << epar1 << endl;
  cout << "ftlin = " << ftlin << " +/- " << eftlin << " kHz" << endl;

  cout << "\n===== FIT LINEARE LOCALE SU phi(f) =====" << endl;
  cout << "intervallo = [" << fitfl0 << ", " << fitfl1 << "] kHz" << endl;
  cout << "parf0 = " << parf0 << " +/- " << eparf0 << endl;
  cout << "parf1 = " << parf1 << " +/- " << eparf1 << endl;
  cout << "ftlinf = " << ftlinf << " +/- " << eftlinf << " kHz" << endl;

  cout<<"freq taglio lineare intorno su phi "<<ftlinf<<endl;
						    
  g->Draw("PSAME");
  gphi->Draw("PSAME");
 
  TLine *lx1 = new TLine(ftlin,0.3,ftlin,0.9);
  lx1->SetLineColor(1);
  lx1->Draw();
 
  TLine *lx0 = new TLine(ftlinf,0.3,ftlinf,0.9);
  lx0->SetLineColor(2);
  lx0->Draw();
 
  TGaxis *axis01 = new TGaxis(1.1*fl1,0,1.1*fl1,1,0,1,510,"+L");
  axis01->SetLineColor(kRed);
  axis01->SetLabelColor(kRed);
  axis01->SetTitle("D#phi (^{#circ}/90)");
  axis01->SetTextColor(kRed);
  axis01->SetLabelSize(0.03);
  axis01->SetTextSize(0.03);
  axis01->Draw();
  
  ll1 = new TLine(fl0,1,fl1,1);
  ll1->SetLineStyle(2);
  ll1->Draw();
 
  ll2 = new TLine(fl0,1./TMath::Sqrt(2.),fl1,1./TMath::Sqrt(2.));
  ll2->SetLineStyle(2);
  ll2->SetLineColor(kBlack);
  ll2->Draw();
 
  ll3 = new TLine(fl0,y0,fl1,y0);
  ll3->SetLineStyle(2);
  ll3->SetLineColor(kRed);
  ll3->Draw();

 
  c00->SaveAs(Form("RCF_4_%s.pdf",lab));
  c00->SaveAs(Form("RCF_4_%s.png",lab)); 

  //return;
  //=============================
 
  TCanvas *c0a = new TCanvas("c0a","c0a",1200,1200);
  c0a->Divide(1,2);
  c0a->cd(1);
  c0a->GetPad(1)->SetLogx();
  c0a->GetPad(2)->SetLogx();

  double residuo[N]={0};
  double eresiduo[N]={0};
  for(int i=0; i<npo; i++){
    eresiduo[i]=eA[i];
    residuo[i]=A[i]-func0->Eval(fr[i]);
  }
  TGraphErrors* gresi = new TGraphErrors(npo,fr,residuo,efr,eresiduo);
  gresi->SetTitle("");
  gresi->GetXaxis()->SetTitle("f (kHz)");
  gresi->GetYaxis()->SetTitle("A (meas.-fit)");
  gresi->SetMarkerStyle(20);
  gresi->Draw("AP");
  TLine *lin = new TLine(xmin,0,xmax,0);
  lin->SetLineStyle(2);
  lin->Draw();

  c0a->cd(2);
  double residup[N]={0};
  double eresidup[N]={0};
  for(int i=0; i<npo; i++){
    eresidup[i]=ephin[i];
    residup[i]=phin[i]-func1->Eval(fr[i]);
  }
  TGraphErrors* gresf = new TGraphErrors(npo,fr,residup,efr,eresidup);
  gresf->SetTitle("");
  gresf->GetXaxis()->SetTitle("f (kHz)");
  gresf->GetYaxis()->SetTitle("#phi/90 (meas.-fit)");
  gresf->SetMarkerStyle(24);
  gresf->SetMarkerColor(kRed);
  gresf->SetLineColor(kRed);
  gresf->Draw("AP");
  lin->SetLineStyle(2);
  lin->Draw();

  c0a->SaveAs(Form("RCF_5_%s.pdf",lab));
  c0a->SaveAs(Form("RCF_5_%s.png",lab)); 

 
  //===========================================================================
 
  TCanvas *c4 = new TCanvas("c4","c4",800,800);
  c4->Divide(1,2);
  c4->cd(1);
 
  const int NN=6;
  double val[NN]={0};
  double eval[NN]={0};
  double myxx[NN]={1,2,3,4,5,6};
  double myex[NN]={0};
  double errel[NN]={0};
  
  val[0]=ftmeas;
  val[1]=ftfit;
  val[2]=ftlin;
 
  val[3]=ftmeas1;
  val[4]=ftfitf;
  val[5]=ftlinf;
 
  eval[0]=eftmeas;
  eval[1]=eftfit;
  eval[2]=eftlin;
 
  eval[3]=eftmeas1;
  eval[4]=eftfitf;
  eval[5]=eftlinf;

  cout<<"CICCIO "<<val[0]<<endl;
    
  for(int i=0; i<NN; i++){
    errel[i]=100*eval[i]/val[i];
    cout<<"metodo "<<i<<" err. rel.: "<<errel[i]<<endl;
  }

 
  TGraphErrors *gout = new TGraphErrors(NN,myxx,val,myex,eval);
  gout->SetMinimum(0.9);
  gout->SetMaximum(1.6);
  gout->SetMarkerStyle(24);
  gout->SetTitle("");
  gout->GetXaxis()->SetTitle("metodo");
  gout->GetYaxis()->SetTitle("f_{t} (kHz)");
  gout->Draw("AP");
  //gout->Fit("pol0");

  TLine* lm = new TLine(0.5,ftexp/1e3,6.5,ftexp/1e3);
  TLine* lm0 = new TLine(0.5,(ftexp-eftexp)/1e3,6.5,(ftexp-eftexp)/1e3);
  TLine* lm1 = new TLine(0.5,(ftexp+eftexp)/1e3,6.5,(ftexp+eftexp)/1e3);
  lm->SetLineColor(kBlue);
  lm->SetLineStyle(2);
  lm0->SetLineColor(kBlue);
  lm0->SetLineStyle(3);
  lm1->SetLineColor(kBlue);
  lm1->SetLineStyle(3);
  lm->Draw();
  lm0->Draw();
  lm1->Draw();
 
  TLegend* myleg = new TLegend(0.7,0.7,0.99,0.99);
  if(opt==1)myleg->AddEntry(gout,"1: A^{2} vs f^{2}","P");
  if(opt==2)myleg->AddEntry(gout,"1: A^{-2} vs f^{-2}","P");

  myleg->AddEntry(gout,"2: #chi^{2} A vs f","P");
  myleg->AddEntry(gout,"3: A intorno","P");
 
  if(opt==1)myleg->AddEntry(gout,"4: tan D#phi vs f","P");
  if(opt==2)myleg->AddEntry(gout,"4: cotan D#phi vs f","P");
  if(opt==1)myleg->AddEntry(gout,"5: #chi^{2} tan D#phi vs f","P");
  if(opt==2)myleg->AddEntry(gout,"5: #chi^{2} cotan D#phi vs f","P");
  myleg->AddEntry(gout,"6: D#phi intorno","P");
 
  myleg->SetLineColor(0);
  myleg->SetLineWidth(0);
  myleg->Draw();
 
  c4->cd(2);
 
  double valC[NN]={0};
  double evalC[NN]={0};
  for(int i=0; i<NN; i++){
    valC[i]=1e9/(2*TMath::Pi()*Rmeas*val[i]*1e3);
    evalC[i]=valC[i]*TMath::Sqrt(pow(eval[i]/val[i],2)+pow(eRmeas/Rmeas,2));
    cout<<"metodo "<<i<<" err. rel.: "<<errel[i]<<endl;
    if(i==2)cout<<"**** INTORNO LIN A "<<valC[i]<<" "<<evalC[i]<<endl;
    if(i==5)cout<<"**** INTORNO LIN Dphi "<<valC[i]<<" "<<evalC[i]<<endl; 
  }



 
  TGraphErrors *goutC = new TGraphErrors(NN,myxx,valC,myex,evalC);
  goutC->SetMinimum(85);
  goutC->SetMaximum(110);
  goutC->SetMarkerStyle(24);
  goutC->SetTitle("");
  goutC->GetXaxis()->SetTitle("metodo");
  goutC->GetYaxis()->SetTitle("C (nF)");
  goutC->Draw("AP");
 
  TLine* lmC = new TLine(0.5,Cmeas*1e9,6.5,Cmeas*1e9);
  TLine* lmC0 = new TLine(0.5,1e9*(Cmeas-eCmeas),6.5,1e9*(Cmeas-eCmeas));
  TLine* lmC1 = new TLine(0.5,1e9*(Cmeas+eCmeas),6.5,1e9*(Cmeas+eCmeas));
  lmC->SetLineColor(kBlue);
  lmC->SetLineStyle(2);
  lmC0->SetLineColor(kBlue);
  lmC0->SetLineStyle(3);
  lmC1->SetLineColor(kBlue);
  lmC1->SetLineStyle(3);
  lmC->Draw();
  lmC0->Draw();
  lmC1->Draw();
  
  c4->SaveAs(Form("RCF_6_%s.pdf",lab));
  c4->SaveAs(Form("RCF_6_%s.png",lab));
 
}
