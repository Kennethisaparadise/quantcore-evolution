import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { FileText, FolderOpen, Plus, Trash2, Edit, Copy, Download, Search } from 'lucide-react';
import { useDocuments } from '@/hooks/useDocuments';
import { db } from '@/lib/local-db';

interface Template {
  id: string;
  name: string;
  type: string;
  content: string;
  category: string;
  created_at: string;
}

const defaultTemplates: Template[] = [
  {
    id: 'template-ucc1',
    name: 'UCC-1 Standard',
    type: 'ucc_filing',
    category: 'Secured Party Creditor',
    content: `UCC-1 FINANCING STATEMENT

1. DEBTOR INFORMATION
   Name: [Debtor Full Legal Name]
   Address: [Debtor Address]

2. SECURED PARTY CREDITOR
   Name: [Your Name]
   Address: [Your Address]

3. COLLATERAL
   [Description of collateral]

4. FILING TYPE
   [ ] Original
   [ ] Amendment
   [ ] Continuation

I certify the above is true and correct.`,
    created_at: new Date().toISOString()
  },
  {
    id: 'template-spc-affidavit',
    name: 'SPC Affidavit of Fact',
    type: 'affidavit',
    category: 'Secured Party Creditor',
    content: `AFFIDAVIT OF SECURED PARTY CREDITOR

State of: [State]
County of: [County]

I, [Full Legal Name], being duly sworn, depose and state:

1. I am a Secured Party Creditor with a valid security interest.
2. Security Agreement dated: [Date]
3. Collateral: [Description]
4. All documents are properly filed and perfected.

I certify under penalty of perjury that the foregoing is true.`,
    created_at: new Date().toISOString()
  },
  {
    id: 'template-debt-demand',
    name: 'Debt Validation Demand',
    type: 'demand_letter',
    category: 'Debt Management',
    content: `DEBT VALIDATION DEMAND

[Creditor Name]
[Address]

Date: [Date]

RE: Account [Account Number]

Dear [Creditor Name],

This is a formal demand for validation of the debt you claim I owe.

Under the Fair Debt Collection Practices Act (FDCPA) and related laws, you must provide:

1. Proof of the original debt agreement
2. Chain of title showing assignment to you
3. Original signed contract or agreement
4. Statement showing the current balance
5. Name and address of the original creditor

Until you provide this documentation, I dispute this debt and demand you cease all collection efforts.

Please provide the requested documentation within 30 days.

Sincerely,
[Your Name]`,
    created_at: new Date().toISOString()
  },
  {
    id: 'template-trust-certificate',
    name: 'Trust Certificate',
    type: 'trust_document',
    category: 'Trusts',
    content: `CERTIFICATE OF TRUST

[Trust Name]

This certifies that:

1. Trust was established on [Date]
2. Trustee: [Trustee Name]
3. Trust Type: [Revocable/Irrevocable]
4. Beneficiaries: [Beneficiary Names]

The Trustee is authorized to open bank accounts, transfer assets, and manage trust property on behalf of the trust.

This certificate is issued for the purpose of [Purpose - e.g., opening bank account].

Trustee Signature: ________________
Date: _____________`,
    created_at: new Date().toISOString()
  },
  {
    id: 'template-notice-credors',
    name: 'Creditor Notice',
    type: 'notice',
    category: 'Trusts',
    content: `NOTICE TO CREDITORS

[Trust/Estate Name]

NOTICE IS HEREBY GIVEN that [Trustee/Executor Name] has been appointed as Trustee/Executor.

All creditors of said trust/estate must file their claims within [Time Period] of this notice.

Claims should be filed with:
[Trustee/Executor Name]
[Address]
[Phone]

This notice published pursuant to state law.`,
    created_at: new Date().toISOString()
  },
  {
    id: 'template-rescission',
    name: 'Contract Rescission',
    type: 'legal_action',
    category: 'General',
    content: `NOTICE OF RESCISSION

[Other Party Name]
[Address]

Date: [Date]

RE: Contract/Agreement dated: [Date]

Dear [Party Name],

This letter serves as formal NOTICE OF RESCISSION of the above-referenced agreement.

Under [Applicable Law - e.g., Truth in Lending Act, cooling off period provisions], I exercise my right to rescind this contract.

All obligations under this agreement are hereby terminated.
Any payments made shall be returned within [Time Period].

This rescission is effective immediately upon receipt of this notice.

Sincerely,
[Your Name]`,
    created_at: new Date().toISOString()
  }
];

export function TemplateManager() {
  const [templates, setTemplates] = useState<Template[]>(defaultTemplates);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [editingTemplate, setEditingTemplate] = useState<Template | null>(null);

  const categories = ['all', ...new Set(templates.map(t => t.category))];

  const filteredTemplates = templates.filter(template => {
    const matchesSearch = template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          template.content.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const handleDuplicate = (template: Template) => {
    const duplicate: Template = {
      ...template,
      id: `template-${Date.now()}`,
      name: `${template.name} (Copy)`,
      created_at: new Date().toISOString()
    };
    setTemplates([...templates, duplicate]);
  };

  const handleDelete = (id: string) => {
    if (confirm('Delete this template?')) {
      setTemplates(templates.filter(t => t.id !== id));
    }
  };

  const handleSaveTemplate = async (template: Partial<Template>) => {
    if (editingTemplate) {
      // Update existing
      setTemplates(templates.map(t => 
        t.id === editingTemplate.id ? { ...t, ...template, id: t.id } : t
      ));
      setEditingTemplate(null);
    } else {
      // Create new
      const newTemplate: Template = {
        id: `template-${Date.now()}`,
        name: template.name || 'New Template',
        type: template.type || 'general',
        category: template.category || 'General',
        content: template.content || '',
        created_at: new Date().toISOString()
      };
      setTemplates([...templates, newTemplate]);
    }
  };

  const exportTemplates = () => {
    const data = JSON.stringify(templates, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trustylaw-templates-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Document Templates</h2>
          <p className="text-muted-foreground">Manage and organize your legal document templates</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={exportTemplates}>
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
          <Button onClick={() => setEditingTemplate({ id: '', name: '', type: '', category: '', content: '', created_at: '' })}>
            <Plus className="mr-2 h-4 w-4" />
            New Template
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search templates..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        <Select value={selectedCategory} onValueChange={setSelectedCategory}>
          <SelectTrigger className="w-48">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {categories.map(cat => (
              <SelectItem key={cat} value={cat}>
                {cat === 'all' ? 'All Categories' : cat}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Templates Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredTemplates.map(template => (
          <Card key={template.id} className="card-elevated hover:shadow-md transition-shadow">
            <CardHeader className="pb-2">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-primary" />
                  <CardTitle className="text-base">{template.name}</CardTitle>
                </div>
                <Badge variant="outline">{template.category}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground line-clamp-3 mb-4">
                {template.content.substring(0, 150)}...
              </p>
              <div className="flex gap-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="flex-1"
                  onClick={() => setEditingTemplate(template)}
                >
                  <Edit className="mr-1 h-3 w-3" />
                  Edit
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="flex-1"
                  onClick={() => handleDuplicate(template)}
                >
                  <Copy className="mr-1 h-3 w-3" />
                  Copy
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => handleDelete(template.id)}
                >
                  <Trash2 className="h-4 w-4 text-destructive" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Edit/Create Modal */}
      {editingTemplate && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-2xl max-h-[90vh] overflow-auto">
            <CardHeader>
              <CardTitle>{editingTemplate.id ? 'Edit Template' : 'Create Template'}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Template Name</Label>
                  <Input
                    value={editingTemplate.name}
                    onChange={e => setEditingTemplate({...editingTemplate, name: e.target.value})}
                    placeholder="UCC-1 Filing Template"
                  />
                </div>
                <div>
                  <Label>Category</Label>
                  <Input
                    value={editingTemplate.category}
                    onChange={e => setEditingTemplate({...editingTemplate, category: e.target.value})}
                    placeholder="Secured Party Creditor"
                  />
                </div>
              </div>
              <div>
                <Label>Template Content</Label>
                <Textarea
                  value={editingTemplate.content}
                  onChange={e => setEditingTemplate({...editingTemplate, content: e.target.value})}
                  placeholder="Enter template content..."
                  rows={15}
                />
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setEditingTemplate(null)}>
                  Cancel
                </Button>
                <Button onClick={() => handleSaveTemplate(editingTemplate)}>
                  {editingTemplate.id ? 'Save Changes' : 'Create Template'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

export default TemplateManager;
